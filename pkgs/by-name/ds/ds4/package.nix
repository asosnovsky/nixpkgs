{ lib
, stdenv
, fetchFromGitHub
, makeWrapper
, curl
, cacert
, python3
, rocmPackages
, cudaPackages

  # Backend selects the build target / toolchain. One of "cpu" | "rocm" | "cuda".
, backend ? "cpu"

  # CPU microarchitecture for host code, as accepted by `-march=` (e.g.
  # "x86-64-v3", "native"). Upstream defaults to `-march=native`, which makes
  # store paths host-specific and non-reproducible; we override it to an empty
  # flag (baseline ISA, safe to share) unless a target is given.
, cpuTarget ? null

  # CUDA GPU arch, e.g. "sm_89". Defaults to the first real architecture the
  # host platform's cudaPackages support (same source as ollama/koboldcpp;
  # nixpkgs convention is `cudaPackages.flags.realArches`). Override for your
  # GPU, e.g. "sm_120", or "native" to target the build host's GPU.
, cudaArch ? null

  # ROCm GPU target, e.g. "gfx1151". Defaults to the build host's detected GPU
  # target (same source as ollama: rocmPackages.clr), falling back to
  # upstream's default "gfx1151" (Strix Halo) when nothing is detected.
, rocmArch ? null
}:

# DwarfStar (antirez/ds4) — a from-source DeepSeek V4 Flash/PRO local inference
# engine. Upstream is a hand-written Makefile with no `install` target and no
# tagged releases, so we pin a `main` commit and write our own installPhase.
#
# This derivation is *backend-parameterized*: pick the GPU backend with the
# `backend` argument ("cpu" | "rocm" | "cuda"); `ds4-rocm` and `ds4-cuda` are
# wired up in all-packages.nix, following the same variant pattern as
# ollama[-cpu,-rocm,-cuda].
#
# GPU variants must be built on a machine carrying the matching toolchain; the
# CUDA/ROCm kernels are tuned per GPU arch via cudaArch/rocmArch.
#
# To bump: set `rev` to the new `main` HEAD and refresh `hash` (start from
# lib.fakeHash and copy the SRI hash from the build error, or use
# `nix-prefetch-url --unpack https://github.com/antirez/ds4/archive/<rev>.tar.gz`
# then `nix hash to-sri --type sha256 <hash>`).

assert lib.elem backend [ "cpu" "rocm" "cuda" ];

let
  # `hf` CLI used by ds4-download-model for the large/sharded GGUF files
  # (MXFP4, PRO, GLM). huggingface-hub provides the `hf` binary; hf-xet is the
  # optional Xet backend it uses to speed up big transfers.
  pythonEnv = python3.withPackages (
    ps: with ps; [
      huggingface-hub
      hf-xet
    ]
  );

  # ROCm libraries the gfx kernels and link flags (-lhipblas -lhipblaslt) need.
  rocmInputs = [
    rocmPackages.clr # provides hipcc + HIP runtime
    rocmPackages.hipblas
    rocmPackages.hipblas-common # hipblas.h includes hipblas-common/hipblas-common.h
    rocmPackages.hipblaslt
    rocmPackages.rocblas
    rocmPackages.rocwmma # gfx1151 backend uses rocWMMA headers
    rocmPackages.hipcub
    rocmPackages.rocprim
    rocmPackages.rocthrust
    rocmPackages.rocm-runtime
  ];

  # -L<dir> and matching rpath so the linked binaries resolve the ROCm .so's
  # from the store at runtime (upstream assumes /opt/rocm on PATH).
  rocmLibDirs = map (p: "${lib.getLib p}/lib") rocmInputs;
  rocmLinkFlags =
    lib.concatStringsSep " "
      (map (d: "-L${d} -Wl,-rpath,${d}") rocmLibDirs);
  rocmIncludeFlags =
    lib.concatStringsSep " "
      (map (p: "-I${lib.getDev p}/include") rocmInputs);

  # CUDA libs we link against; nvcc is the linker for the cuda build.
  cudaLibDirs = [
    "${lib.getLib cudaPackages.cuda_cudart}/lib"
    "${lib.getLib cudaPackages.libcublas}/lib"
  ];
  cudaLinkFlags =
    lib.concatStringsSep " "
      (map (d: "-L${d} -Xlinker -rpath -Xlinker ${d}") cudaLibDirs);

  # Arch defaults, resolved from the host platform like ollama/koboldcpp do.
  resolvedCudaArch = if cudaArch != null then cudaArch else builtins.head (cudaPackages.flags.realArches or [ "sm_89" ]);
  clrGpuTargets = (rocmPackages.clr.localGpuTargets or [ ]) ++ (rocmPackages.clr.gpuTargets or [ ]);
  resolvedRocmArch = if rocmArch != null then rocmArch else if clrGpuTargets != [ ] then builtins.head clrGpuTargets else "gfx1151";

  # Host-code -march flag; empty (baseline) unless cpuTarget is given. Passed to
  # every backend since upstream's CFLAGS/NVCCFLAGS both embed NATIVE_CPU_FLAG.
  marchFlag = lib.optionalString (cpuTarget != null) "-march=${cpuTarget}";

  # phony Makefile target per backend.
  buildTarget = {
    cpu = "cpu";
    rocm = "strix-halo";
    cuda = "cuda";
  }.${backend};

  # Per-backend command-line variable overrides (forwarded to the recursive
  # sub-make as MAKEOVERRIDES, beating the Makefile's `?=` defaults).
  #
  # `makeFlags` entries are expanded UNQUOTED by the generic builder, so any
  # value containing spaces must go in `backendMakeFlagsArray` instead (see
  # preBuild below).
  backendMakeFlags = {
    cpu = [ ];
    rocm = [ "ROCM_ARCH=${resolvedRocmArch}" ];
    cuda = [
      "NVCC=${lib.getExe' cudaPackages.cuda_nvcc "nvcc"}"
      "CUDA_HOME=${cudaPackages.cuda_nvcc}"
      "CUDA_ARCH=${resolvedCudaArch}"
    ];
  }.${backend};

  # Space-containing variable assignments.
  backendMakeFlagsArray = {
    cpu = [ ];
    rocm = [
      # Append store include/lib paths to the upstream ROCm flags (hipcc is the
      # raw ROCm compiler, not the nix cc-wrapper, so it needs explicit -I/-L).
      "ROCM_CFLAGS=-O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=${resolvedRocmArch} ${rocmIncludeFlags}"
      "ROCM_LDLIBS=-lm -pthread ${rocmLinkFlags} -lhipblas -lhipblaslt -lrocblas"
    ];
    cuda = [
      # Replace upstream's hardcoded /usr/local/cuda + sbsa-linux (aarch64)
      # paths with the nixpkgs cudart/cublas store paths. NVCCFLAGS is
      # overridden rather than patched because upstream embeds
      # `-Xcompiler $(NATIVE_CPU_FLAG)`, which leaves a dangling `-Xcompiler`
      # when NATIVE_CPU_FLAG is empty. NVCC_ARCH_FLAGS is still computed by the
      # Makefile from the CUDA_ARCH we pass above.
      "NVCCFLAGS=-O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS)${lib.optionalString (marchFlag != "") " -Xcompiler ${marchFlag}"} -Xcompiler -pthread"
      "CUDA_LDLIBS=-lm -Xcompiler -pthread ${cudaLinkFlags} -lcudart -lcublas"
    ];
  }.${backend};
in
stdenv.mkDerivation (finalAttrs: {
  pname = "ds4" + lib.optionalString (backend != "cpu") "-${backend}";
  version = "0-unstable-2026-09-16";

  src = fetchFromGitHub {
    owner = "antirez";
    repo = "ds4";
    rev = "8db1d1d155cb0400a86a86b9c62d0defb3a6148b";
    hash = "sha256-d0TRJH5/cNlDrgJy2i9eEUuSAlnka0BvxDXlXIwMwrE=";
  };

  __structuredAttrs = true;
  strictDeps = true;

  enableParallelBuilding = true;

  nativeBuildInputs =
    [ makeWrapper ]
    ++ lib.optionals (backend == "rocm") [ rocmPackages.clr ]
    ++ lib.optionals (backend == "cuda") [ cudaPackages.cuda_nvcc ];

  buildInputs =
    lib.optionals (backend == "rocm") rocmInputs
    ++ lib.optionals (backend == "cuda") [
      cudaPackages.cuda_cudart
      cudaPackages.libcublas
    ];

  # With __structuredAttrs, makeFlags is a list passed verbatim to make (no
  # word splitting), so the space-containing entries above can live here.
  makeFlags =
    [ "NATIVE_CPU_FLAG=${marchFlag}" ]
    ++ backendMakeFlags
    ++ backendMakeFlagsArray;
  buildFlags = [ buildTarget ];

  # CPU-runnable tests only; the GPU variants are validated on real hardware.
  doCheck = backend == "cpu";
  checkPhase = ''
    runHook preCheck
    local flagsArray=()
    concatTo flagsArray makeFlags
    make "''${flagsArray[@]}" tests/test_layer_pack tests/test_gpu_args q4k-dot-test mxfp4-dot-test
    ./tests/test_layer_pack
    ./tests/test_gpu_args
    ./ds4-eval --self-test-extractors
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 -t "$out/bin" ds4 ds4-server ds4-bench ds4-eval ds4-agent

    # Wire upstream's GGUF downloader in as `ds4-download-model`. Patch its
    # project-root detection so the gguf dir and the `ds4flash.gguf` symlink
    # land in a writable location ($DS4_HOME, default: cwd) instead of the
    # read-only store dir that `dirname $0` would resolve to here.
    install -Dm755 download_model.sh "$out/bin/ds4-download-model"
    substituteInPlace "$out/bin/ds4-download-model" \
      --replace-fail 'ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)' 'ROOT=''${DS4_HOME:-$PWD}' \
      --replace-warn './download_model.sh' 'ds4-download-model'
    runHook postInstall
  '';

  # `ds4-download-model` shells out to curl (and optionally the `hf` CLI for the
  # huge PRO files); make curl available and point it at a CA bundle if the
  # environment doesn't already set one.
  postFixup = ''
    wrapProgram "$out/bin/ds4-download-model" \
      --prefix PATH : ${lib.makeBinPath [ curl pythonEnv ]} \
      --set-default SSL_CERT_FILE ${cacert}/etc/ssl/certs/ca-bundle.crt
  '';

  # Only the CPU variant is guaranteed to run without a GPU/device present, so
  # restrict the smoke check to it. GPU variants are validated on real hardware.
  doInstallCheck = backend == "cpu";
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/ds4" --help >/dev/null
    runHook postInstallCheck
  '';

  meta = {
    description = "DeepSeek V4 Flash/PRO local inference engine (DwarfStar)";
    homepage = "https://github.com/antirez/ds4";
    license = lib.licenses.mit;
    sourceProvenance = [ lib.sourceTypes.fromSource ];
    platforms = lib.platforms.linux; # Metal backend is macOS-only, out of scope
    mainProgram = "ds4";
    maintainers = with lib.maintainers; [ asosnovsky ];
  };
})

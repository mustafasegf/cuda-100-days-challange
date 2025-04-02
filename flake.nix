{
  description = "cuda devshell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in with pkgs; {
        devShells.default = mkShell {
          nativeBuildInputs = [ cmake ninja gnumake pkg-config ];
          buildInputs = [ cudaPackages.cudatoolkit ];

          packages = with pkgs; [ gcc12 ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.zluda}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib/:$LD_LIBRARY_PATH
            export CUDA_PATH=${pkgs.cudatoolkit}
          '';
        };
      });
}

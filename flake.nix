{
  description = "Flakes-based nix shell for motilitAI based on micromamba";

  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          name = "motilitAI";

          pkgs = import nixpkgs {
            config = {
              # CUDA and other "friends" contain unfree licenses. To install them, you need this line:
              allowUnfree = true;
            };
            inherit system;
          };
        in
        rec {
          defaultPackage = pkgs.buildFHSUserEnv {
            inherit name;

            targetPkgs = pkgs: with pkgs; [
              micromamba
              jdk
              libGL
            ];
            runScript = "bash";

            profile = ''
              eval "$(micromamba shell hook -s bash)"
              micromamba create -q -n ${name}
              micromamba activate ${name}
              micromamba install --yes -f environment.yml
            '';
          };
        }
      );
}

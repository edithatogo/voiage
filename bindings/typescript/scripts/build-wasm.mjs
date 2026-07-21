import { mkdirSync, renameSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const repository = resolve(here, "../../..");
const output = resolve(here, "../wasm");
const target = "wasm32-unknown-unknown";
const rustc = spawnSync("rustup", ["which", "rustc"], { encoding: "utf8" }).stdout.trim();
const environment = { ...process.env, RUSTC: rustc };

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: repository,
    env: environment,
    stdio: "inherit",
  });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

run("rustup", ["target", "add", target]);
run("rustup", ["run", "stable", "cargo", "build", "--manifest-path", "rust/Cargo.toml", "--release", "--locked", "--target", target, "--package", "voiage-wasm"]);
mkdirSync(output, { recursive: true });
run("wasm-bindgen", [
  "--target",
  "nodejs",
  "--out-dir",
  output,
  resolve(repository, "rust/target/wasm32-unknown-unknown/release/voiage_wasm.wasm"),
]);
renameSync(resolve(output, "voiage_wasm.js"), resolve(output, "voiage_wasm.cjs"));

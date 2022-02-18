# Common Processing Blocks

[![Continuous integration](https://github.com/hotg-ai/proc-blocks/workflows/Continuous%20integration/badge.svg?branch=master)](https://github.com/hotg-ai/proc-blocks/actions)

([API Docs])

Processing blocks built by [Hammer of the Gods][hotg] that you can use with your
Runes.

## For Developers

### Releasing

Whenever the `hotg-rune-proc-blocks` and `hotg-rune-core` crates make a semver
breaking release you will need to bump their version numbers in each proc
block's `Cargo.toml` file and fix any compile errors.

Afterwards, use [`cargo-release`][cargo-release] to update all version numbers
and tag the commit appropriately.

```console
$ cargo release --workspace 0.11.1
```

### Metadata

When loaded by a WebAssembly runtime, a processing block will automatically
provide the caller with information about itself.

This metadata may include

- The name and version number
- A human-friendly description
- Supported arguments
- Input and output tensors

To access this metadata, compile the proc block to WebAssembly and make sure
to activate the `metadata` feature.

```console
$ cd argmax
$ cargo build --target wasm32-unknown-unknown --features metadata
...
   Compiling hotg-rune-core v0.11.3
   Compiling hotg-rune-proc-block-macros v0.11.3
   Compiling hotg-rune-proc-blocks v0.11.3
   Compiling argmax v0.11.3
    Finished dev [unoptimized + debuginfo] target(s) in 8.78s

$ ls ../target/wasm32-unknown-unknown/debug/ -l
.rw-r--r--  312 consulting 18 Feb 23:18 argmax.d
.rwxr-xr-x 2.5M consulting 18 Feb 23:18 argmax.wasm
drwxr-xr-x    - consulting 18 Feb 23:18 build
drwxr-xr-x    - consulting 18 Feb 23:18 deps
drwxr-xr-x    - consulting 18 Feb 23:18 examples
drwxr-xr-x    - consulting 18 Feb 23:18 incremental
.rw-r--r--  315 consulting 18 Feb 23:18 libargmax.d
.rw-r--r-- 420k consulting 18 Feb 23:18 libargmax.rlib
```

These WebAssembly modules are fairly hefty out of the box due to all the debug
information inside.

To help with this, we've created a helper script that will compile all proc
blocks in the repository to WebAssembly and strip out debug information.

```console
$ RUST_LOG=warn,xtask=info cargo xtask dist --out-dir target/proc-blocks
    Finished dev [unoptimized + debuginfo] target(s) in 0.04s
     Running `target/debug/xtask dist --out-dir target/proc-blocks`
 INFO Compile: xtask::build: Compiling proc-blocks to WebAssembly
    Finished release [optimized] target(s) in 0.03s
 INFO xtask: Stripping custom sections to reduce binary size
 INFO xtask: Creating the release bundle

$ ls -la target/proc-blocks
...
.rw-r--r--  16k consulting 18 Feb 23:22 argmax.wasm
.rw-r--r--   19 consulting 18 Feb 23:22 manifest.json
.rw-r--r-- 2.6k consulting 18 Feb 23:22 metadata.json
```

Besides the compiled binaries, there are also two files

- `manifest.json` - a list of each `*.wasm` file that was compiled
- `metadata.json` - a serialized version of the metadata that has been extracted
  from each processing block

```console
$  cat target/proc-blocks/manifest.json
[
  "argmax.wasm"
]

$ cat target/proc-blocks/metadata.json
{
  "argmax.wasm": {
    "name": "argmax",
    "version": "0.11.3",
    "description": "",
    "repository": "",
    "tags": ["max", "numeric"],
    "arguments": [],
    "inputs": [
      {
        "name": "input",
        "hints": [
          {
            "type": "example-shape",
            "value": { "element_type": "u8", "dimensions": { "type": "dynamic" } }
          },
          ...
        ]
      }
    ],
    "outputs": [
      {
        "name": "max",
        "description": "The index of the element with the highest value",
        "hints": []
      }
    ]
  }
}
```

> **Note:** the `metadata.json` file is provided as a convenience for
> troubleshooting purposes. The precise format may change without warning and
> shouldn't be relied on.

## License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE.md) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT.md) or
   http://opensource.org/licenses/MIT)

at your option.

It is recommended to always use [cargo-crev][crev] to verify the
trustworthiness of each of your dependencies, including this one.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

The intent of this crate is to be free of soundness bugs. The developers will
do their best to avoid them, and welcome help in analysing and fixing them.

[API Docs]: https://hotg-ai.github.io/proc-blocks
[crev]: https://github.com/crev-dev/cargo-crev
[hotg]: https://hotg.dev/
[cargo-release]: https://crates.io/crates/cargo-release

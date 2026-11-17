# Coin CBC Rust bindings

Rust bindings to the CoinOR CBC MILP Solver using the C API.

Tested on Debian 10, AMD64, coinor-libcbc3 2.9.9+repack1-1.
For more details on installing the `libCbc` dependencies, [see below](#prerequisites-installing-cbc-library-files).

## `coin_cbc_sys`

This crate exposes raw bindings to the C functions.

## `coin_cbc`

This crate exposes safe rust bindings using `coin_cbc_sys`.
`coin_cbc::raw::Model` exposes direct translation of the C function with assert to guaranty safe use.
`coin_cbc::Model` exposes a more user friendly, rustic and efficient API: it was used successfully to solve MILP with 250,000 binary variables with unnoticeable overhead.

## Examples

See the [examples directory](examples/).

## Prerequisites: installing `Cbc` library files

The library files of the [`COIN-OR` Solver `Cbc`](https://github.com/coin-or/Cbc) need to present on your system when compiling a project that depends on `coin_cbc`.
On a Debian system with a user with admin rights, this is easily achieved with:
```
sudo apt install coinor-libcbc-dev
```

If you have `pkg-config` available, it'll be used to locate the library.

For other systems, without admin rights or if you need a newer version of `Cbc` (e.g. with bug fixes), you can install `Cbc` through `coinbrew`:
https://coin-or.github.io/user_introduction#building-from-source

You will then have to either:
1. register the resulting library files with your system, or 
2. provide `cargo` with the location of that library.
For the first option, `coinbrew` provides a command suggestion after successful compilation.
The second option can e.g. be done via:
```
RUSTFLAGS='-L /path/to/your/cbc/install/lib' cargo test
```

## Solving multiple problems in parallel

By default, this crate enforces a global lock which will force multiple
problems to be solved sequentially even if `solve` is called from multiple
threads in parallel. This is because by default, libcbc is not thread safe.
If you have compiled your own libcbc with the `CBC_THREAD_SAFE` option,
you can disable this behavior by disabling the `singlethread-cbc`
feature on this crate. Do not disable this feature if you are not certain 
that you have a thread safe libcbc, or you will be exposed to memory corruption
vulnerabilities.

## License

This project is distributed under the [MIT License](LICENSE) by
[Kardinal](https://kardinal.ai).

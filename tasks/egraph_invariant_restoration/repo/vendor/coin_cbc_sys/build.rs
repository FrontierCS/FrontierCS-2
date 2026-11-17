extern crate pkg_config;

fn main() {
    let _ = pkg_config::probe_library("cbc");
}

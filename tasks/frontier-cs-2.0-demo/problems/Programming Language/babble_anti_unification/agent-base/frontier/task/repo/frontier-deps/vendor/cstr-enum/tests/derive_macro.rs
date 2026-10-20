#[test]
fn compile_tests() {
  let cases = trybuild::TestCases::new();
  cases.pass("tests/compilation/pass.rs");
  cases.compile_fail("tests/compilation/duplicate_arg.rs");
  cases.compile_fail("tests/compilation/wrong_arg.rs");
  cases.compile_fail("tests/compilation/non_unit_variant_fail.rs");
  cases.pass("tests/compilation/non_unit_variant_pass.rs");
  cases.compile_fail("tests/compilation/non_enum.rs");
  cases.compile_fail("tests/compilation/name_nul_bytes.rs");
  cases.compile_fail("tests/compilation/name_on_enum.rs");
}

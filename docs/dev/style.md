# Style

The rules for writing FrontierCS 2's code: Python, shell and Make.

- PREFER parsing input once at the boundary (`task_manifest` for `task.toml`, the site schema
  for a site configuration) and passing the parsed values everywhere past it; code past the
  parser does not check again what parsing established
- PUT a format's parser in one module and import it; never copy a format's rules into a
  consumer
- PREFER interfaces (flags, configuration keys, `task.toml` fields) that express exactly what
  real runs need: a value fixed in every real run is a constant, and each setting has one way
  to set it
- NAME a number by what it measures, its unit last (`timeout_seconds`, `payload_bytes`)
- BOUND whatever input or time can grow, each read, retry, wait and subprocess, by a named
  limit at its home
- FAIL loudly: raise with a message on any state the code does not expect; never add a
  fallback, a compatibility shim, a skip flag or a defensive re-check of what a caller already
  guarantees
- REMOVE an old path in the change that replaces it; no deprecated aliases, wrappers or legacy
  branches stay behind
- WRITE error messages as what is wrong, then what is expected
  (`task.toml has no [submission]: declare the editable paths`)
- KEEP frozen algorithms frozen: the source identity, `stable_ulid` and the canonical JSON
  hashers define identities already published; changing one is a migration, not a refactor
- PIN every image by digest and every tool by version and hash; never overwrite a published
  tag with different content
- NEVER commit credentials, tokens, Terraform state or a site's own configuration
  (`config/site.json`)

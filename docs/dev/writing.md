# Writing

The rules for FrontierCS 2's docstrings, comments, docs, commit messages and pull request texts.

- WRITE each rule or invariant once, at its home (the module, the function, the `task.toml`
  field in `tasks/template/TEMPLATE.md`); every other doc links to it or names it
- WRITE a module docstring that says what the module is for and where its rules live; it never
  walks through an algorithm
- WRITE the why of a state in a comment at that code; a doc or a comment never narrates history
  or the conversation that produced the code
- NEVER restate what a name or a signature says
- KEEP files self-contained: no issue numbers, private paths, people's names or links to
  records outside the repository in code, comments or docs; context of that kind belongs in
  the pull request
- PREFER a check, a test or a fixture that pins a format to prose that describes it
- WRITE commit subjects in lowercase imperative, with bodies that explain what and why to a
  reader without other context
- WRITE pull request texts with a short "Why" and "What changes"; say what behaviour changes,
  if any, and how you verified it
- ONE concern per pull request, nothing speculative; a new dependency or a new module is an
  architectural change, so say why in the pull request
- UPDATE `ARCHITECTURE.md` in the same pull request only when a module boundary, or an
  invariant it states, changes

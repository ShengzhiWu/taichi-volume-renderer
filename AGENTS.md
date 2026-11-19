# Repository Collaboration Rules

- When the user asks to add a feature, parameter, keyboard shortcut, or anything else, first check the repository for an identical or similar implementation. If one exists, tell the user where to find it, how to use it, and how it differs from the request. Avoid adding duplicates.
- Skip smoke tests by default.
- Use placeholder paths in all command examples in the repository. Omit actual filenames of local test scenes.
- Avoid parallel implementations where practical. Keep separate implementations if sharing code would increase the total amount of code, introduce unnecessary abstraction, reduce performance, or complicate interfaces.
- If a dependency, file, repository, tool, or entry point needed for the requested workflow is missing or cannot be located, tell the user explicitly what is missing and which step it blocks before proceeding with any fallback. Label any fallback result clearly and state which requested step still needs to be completed or verified.
- Write commit subjects as `<type>: <summary>`. Use a lowercase type that matches the change, such as `feat`, `docs`, `perf`, `chore`, or `fix`.

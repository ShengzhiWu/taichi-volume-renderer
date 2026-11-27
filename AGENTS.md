# Repository Collaboration Rules

- For questions or investigation requests, inspect and report your findings. Changes to files, code, or project state require an explicit user request.
- Interpret a question about a scene's resolution as asking for the model's resolution. Answer only with the model's resolution.
- When the user asks to add a feature, parameter, keyboard shortcut, or anything else, first check the repository for an identical or similar implementation. If one exists, tell the user where to find it, how to use it, and how it differs from the request. Avoid adding duplicates.
- Skip smoke tests by default.
- Use placeholder paths in all command examples in the repository. Omit actual filenames of local test scenes.
- Create Git branches or worktrees only when the user explicitly requests their creation.
- Extend the canonical implementation through the existing code path. Always report any proposed parallel workflow, implementation, or fork to the user before building it. This applies to every task and component, including experiments, subclasses, and scripts inside or outside the repository. Explain what would run in parallel, why the existing path is insufficient, and the substantial reason for the change. Obtain the user's explicit approval before implementing it.
- Explicitly report every change to a rendering algorithm, including work in experiments outside the repository. Before implementation, explain the current behavior, the proposed algorithmic change, why it is needed, and its expected impact on rendering. In the final report, state what changed and how it was verified.
- Before implementing any substantial change that the user's wording does not explicitly include, report it separately and obtain explicit confirmation. This applies even when you judge the change to be an inevitable consequence of the request. Describe the change, why you believe it is required, and its expected effects.
- If a dependency, file, repository, tool, or entry point needed for the requested workflow is missing or cannot be located, tell the user explicitly what is missing and which step it blocks before proceeding with any fallback. Label any fallback result clearly and state which requested step still needs to be completed or verified.
- Write commit subjects as `<type>: <summary>`. Use a lowercase type that matches the change, such as `feat`, `docs`, `perf`, `chore`, or `fix`.
- Whenever the user asks for a commit, include any current changes to `AGENTS.md` in that same commit.
- Use the project defaults unless the user requests a different value or there is a concrete reason to override a default. State the reason when you choose an override.
- Use the most straightforward implementation for every change. If a requested change needs a more complex design, explain the design and why the straightforward approach is insufficient, then obtain the user's explicit confirmation before implementing it.
- Fix renderer defects in the renderer. Keep the original scene data and settings as the reference when diagnosing or verifying a renderer fix. Do not change geometry, materials, lighting, or other scene inputs to conceal a renderer defect.
- Pass all generated scene data to the renderer through one scene file. Do not pass scene data through sidecar files, imported scene-generation code, or any other channel. Rendering settings, lights, and special material attributes may be supplied directly at render time, with a combined maximum of 30 scalar values (floats, integers, or booleans). Count each component of a vector, matrix, or collection separately, and use these settings only for their stated purpose.
- When designing any scene, use material emission only for objects that emit light in reality.

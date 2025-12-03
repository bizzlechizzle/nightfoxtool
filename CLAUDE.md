# Development Guidelines

Universal principles for AI-assisted development.

## Development Rules

1. **Scope Discipline** — Only implement what the current request describes; no surprise features
2. **Prefer Open Source + Verify Licenses** — Default to open tools, log every dependency license
3. **Offline-First** — Assume zero connectivity; add graceful fallbacks when online helpers exist
4. **One Script = One Function** — Keep each script focused, under ~300 lines
5. **No AI in Docs** — Never mention Claude, ChatGPT, Codex, or similar in user-facing docs or UI
6. **Keep It Simple** — Favor obvious code, minimal abstraction, fewer files
7. **Binary Dependencies Welcome** — Freely add binaries (ffmpeg, exiftool, etc.) when they solve problems better than pure-JS alternatives

## Do Not

- Invent new features beyond what the task authorizes
- Leak or transmit local data outside the user's machine
- Add third-party SDKs without logging licenses and confirming they function offline
- Mention AI assistants in UI, user docs, exports, or metadata
- Leave TODOs or unexplained generated code in production branches
- **Assume when uncertain** — If a task is ambiguous or conflicts with these rules, stop and ask

## Stop and Ask When

- Task conflicts with a rule in this file
- Referenced file or path doesn't exist
- Task scope is unclear or seems to exceed "one feature"
- You're about to delete code without understanding why it exists

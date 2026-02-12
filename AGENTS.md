# AGENTS.md — Agentic Coding Guidelines

> This file provides context and conventions for AI coding agents operating in this repository.

## Repository Overview

This repository is currently a greenfield project. The following guidelines establish conventions to be followed as the codebase grows.

## Build & Development Commands

### Primary Commands
```bash
# Install dependencies
npm install

# Development server
npm run dev

# Production build
npm run build

# Run all tests
npm test

# Run a single test file
npm test -- path/to/test.spec.ts

# Run tests matching a pattern
npm test -- --grep "pattern"

# Lint check
npm run lint

# Lint fix
npm run lint:fix

# Type check
npm run typecheck
```

## Code Style Guidelines

### Language & Types
- Use TypeScript for all new code
- Enable strict mode in tsconfig.json
- Avoid `any` — use `unknown` with type guards when types are uncertain
- Prefer `interface` over `type` for object shapes
- Use explicit return types on public functions

### Naming Conventions
- **Files**: kebab-case.ts (e.g., `user-service.ts`)
- **Classes**: PascalCase (e.g., `UserService`)
- **Functions**: camelCase (e.g., `getUserById`)
- **Constants**: SCREAMING_SNAKE_CASE for true constants (e.g., `MAX_RETRY_COUNT`)
- **Types/Interfaces**: PascalCase (e.g., `UserProfile`)
- **Enums**: PascalCase for name, PascalCase for members (e.g., `Status.Active`)
- **Private members**: prefix with underscore (e.g., `_internalCache`)
- **Boolean variables**: prefix with is/has/should (e.g., `isLoading`, `hasPermission`)

### Imports & Organization
- Group imports: external → internal → relative
- Sort imports alphabetically within groups
- Use path aliases (e.g., `@/utils/logger`) over relative imports when available
- Avoid barrel exports for large modules (can hurt tree-shaking)

### Formatting
- Use 2 spaces for indentation
- Max line length: 100 characters
- Use single quotes for strings
- Trailing commas in multi-line objects/arrays
- Semicolons required

### Error Handling
- Use specific error classes over generic Errors
- Always include error context (e.g., `throw new ValidationError("email", "Invalid format")`)
- Handle async errors with try/catch or .catch()
- Never swallow errors — log or re-throw
- Prefer early returns to deep nesting

### Testing
- Use descriptive test names: `should {expected behavior} when {condition}`
- Follow AAA pattern: Arrange → Act → Assert
- One assertion per test (ideally)
- Use factories/test data builders, not hardcoded mocks
- Mock external dependencies, not the code under test

### Git Conventions
- Use conventional commits: `type(scope): description`
- Types: feat, fix, docs, style, refactor, test, chore
- Write imperative commit messages: "Add feature" not "Added feature"
- Keep commits atomic and focused

## AI Agent Instructions

### Before Making Changes
1. Read existing code to understand patterns
2. Check for similar implementations to follow
3. Run existing tests to ensure they pass
4. Identify the minimal change needed

### While Making Changes
1. Match existing code style exactly
2. Add/update tests for changed functionality
3. Update documentation if behavior changes
4. Run lint and typecheck before finishing

### After Making Changes
1. Run tests: `npm test -- path/to/changed/file`
2. Verify build: `npm run build`
3. Check types: `npm run typecheck`
4. Run lint: `npm run lint`

## Decision Log

| Date | Decision | Context |
|------|----------|---------|
| 2026-02-12 | Initial conventions | Greenfield project setup |

## External References

- Cursor rules: `.cursor/rules/` or `.cursorrules`
- Copilot instructions: `.github/copilot-instructions.md`

---

*Last updated: 2026-02-12*

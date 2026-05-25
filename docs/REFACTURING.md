# Best practices for refactoring large codebases with Codex

## Executive summary

Use Codex as a **strictly guided refactoring agent**, not as a “clean up the whole repo” generator.

The safest pattern is:

1. **Set repo-level rules first.**
2. **Map before editing.**
3. **Refactor one responsibility or one module boundary at a time.**
4. **Keep public APIs and old import paths working.**
5. **Run the smallest meaningful validation after every pass.**
6. **Review the diff for behavior drift, removed logs, broken contracts, and unrelated churn.**
7. **Commit one safe pass, then repeat.**

The main rule:

> **Codex should not finish the whole refactor. Codex should finish the next safe refactor pass.**

A stronger version from Take 2 is worth adopting:

> **Codex may only refactor as far as your smallest hard parity gate can still prove behavior stayed the same.**

Where the two takes differ, Take 2 gives the broader production operating model, while Take 1 gives the cleaner day-to-day workflow. The best combined approach is: use Take 2’s larger system around Codex, but keep Take 1’s discipline of small, reviewable, one-responsibility changes.

---

## 1. Set up Codex guardrails before any refactor

### Use `AGENTS.md` as policy, not as a giant manual

Take 1 says to add or update `AGENTS.md` at the repo root. Take 2 adds a useful refinement: keep `AGENTS.md` short and use it as a table of contents or policy layer, while deeper architecture and migration notes live under `docs/`.

The combined recommendation:

```text
AGENTS.md
docs/
  ARCHITECTURE.md
  REFACTURING.md
  module-map.md
  api-compatibility.md
  exec-plans/
    active/
    completed/
  rollout-playbooks/
.agents/
  skills/
    refactor-split/
      SKILL.md
      scripts/
.codex/
  config.toml
```

For this repository, treat `docs/ARCHITECTURE.md`, `docs/module-map.md`, and `docs/api-compatibility.md` as the required starting context before planning a refactor pass.

Use root-level `AGENTS.md` for global rules. Use nested instructions or repo-specific docs when one part of the codebase has special rules.

### Recommended `AGENTS.md` refactoring rules

```md
## Refactoring rules

- Refactors must preserve existing behavior unless explicitly requested.
- Do not remove log statements, metrics, tracing, analytics, warnings, audit events, comments explaining business rules, error handling, feature flags, compatibility shims, or telemetry without explicit approval.
- Keep public APIs, imports, exports, routes, serialized formats, CLI flags, config keys, environment variables, error types, and error messages backward compatible.
- Split large files one responsibility at a time.
- Do not combine refactoring with dependency upgrades, framework migrations, formatting-only rewrites, or behavior changes.
- Before editing, state:
  1. current behavior,
  2. proposed structural change,
  3. public API compatibility plan,
  4. validation commands.
- After editing, run the smallest relevant test, then lint/typecheck if applicable.
- If a public path moves, add a deprecated re-export, wrapper, or facade.
- If validation fails, fix only failures caused by this refactor.
```

This turns Codex’s task from “make the code better” into “make this one structural improvement without changing behavior.”

### Always create a checkpoint

Before each pass:

```bash
git status
git add .
git commit -m "checkpoint before refactor"
```

Or stash if the work is temporary:

```bash
git stash push -m "checkpoint before refactor"
```

Take 1 emphasizes this because Codex can modify many files. Take 2 expands the same idea into worktrees for parallel workstreams.

Use one worktree per major refactor stream when work is large:

```bash
git worktree add ../repo-billing-refactor -b refactor/billing-boundaries
git worktree add ../repo-user-refactor -b refactor/user-service-split
```

That keeps unrelated Codex work from colliding.

---

## 2. Use Codex as a planner before using it as an editor

Do **not** start with:

```text
Clean up this file.
```

That is too vague. It invites broad, risky changes.

Start with a no-edit map.

### No-edit responsibility map prompt

```text
/plan

Analyze @path/to/LargeFile only. Do not edit.

Identify Single Responsibility Principle violations by listing:
- each responsibility this file currently owns,
- methods, functions, or classes involved,
- public APIs exposed,
- side effects: logging, metrics, tracing, IO, DB calls, network calls, queues, feature flags,
- tests that cover each responsibility,
- missing tests or weak coverage,
- safest extraction candidates.

Rank extraction candidates from lowest risk to highest risk.

For each candidate, propose:
- one small class, function, or module to extract,
- target file path,
- public API compatibility plan,
- validation command that proves behavior stayed stable.
```

Codex should first produce an **extraction matrix**, not a patch.

Example matrix shape:

| Responsibility | Code involved | Side effects | Public API risk | Suggested extraction | Validation |
|---|---|---:|---:|---|---|
| Email normalization | `register`, `normalizeEmail` | None | Low | pure function | focused unit test |
| Registration logging | `register` | Logs | Medium | log helper or keep in orchestrator | log contract test |
| Identity creation | `register`, `AuthClient` | Network | High | gateway wrapper later | integration/contract test |
| User persistence | `register`, `UserRepo` | DB | High | repository boundary later | repository tests |
| Welcome email | `register`, `Mailer` | External side effect | High | notification service later | mock side-effect test |

---

## 3. Find SRP violations with both metrics and Codex analysis

Take 2 adds a useful production practice: combine static signals with semantic analysis.

Codex is good at identifying hidden responsibilities, side effects, and risky paths. Metrics are good at finding hotspots quickly.

### Useful hotspot signals

| Signal | What it usually means | Example tools mentioned in the takes |
|---|---|---|
| Very long file | Multiple responsibilities have accumulated | ESLint `max-lines`, detekt `LargeClass`, PMD `NcssCount` |
| Long function | Logic should likely be split | ESLint `max-lines-per-function`, detekt `LongMethod` |
| High complexity | Too many decisions or mixed flows | Sonar Cognitive Complexity, Ruff `C901` |
| Many parameters | Too much orchestration or weak boundaries | ESLint `max-params` |
| Many statements | Too much work in one scope | ESLint `max-statements` |
| Many dependencies | Domain, IO, logging, and framework code are mixed | dependency-cruiser, ArchUnit, Import Linter |

Do not let metrics decide the refactor alone. Use them to choose where Codex should inspect first.

---

## 4. Break up large files one responsibility at a time

The extraction order matters.

A safe order is:

1. **Pure helpers with no side effects**
2. **Validation and parsing**
3. **Formatting and serialization**
4. **Value objects**
5. **IO gateways**
6. **Business rules**
7. **Notifications, queues, external integrations**
8. **Orchestration and control flow last**

The orchestrator file should shrink slowly. It can keep the same public class or function while delegating to extracted classes internally.

### One-responsibility extraction prompt

```text
Extract only the <responsibility name> responsibility from @path/to/LargeFile into <NewClassOrModule>.

Constraints:
- No behavior changes.
- Preserve all log statements, log levels, message text, structured log fields, metrics, tracing, exception types, exception messages, return values, ordering, retries, timeouts, and feature-flag behavior.
- Keep all existing public APIs, exports, and imports working.
- Do not rename public methods.
- Do not reformat unrelated code.
- Do not introduce new dependencies.
- Add adapter, re-export, wrapper, or facade code if needed for backward compatibility.
- Update or add tests only to capture existing behavior.

Before editing, state:
1. current behavior,
2. exact code to move,
3. compatibility plan,
4. tests/checks to run.

After editing:
- run those checks,
- summarize the diff,
- list remaining risk.
```

### Example: safe first extraction

Starting point:

```ts
export class UserService {
  constructor(
    private readonly repo: UserRepo,
    private readonly auth: AuthClient,
    private readonly mailer: Mailer,
    private readonly logger: Logger,
  ) {}

  async register(input: RegisterInput): Promise<UserDto> {
    if (!input.email.includes("@")) throw new Error("invalid_email");

    this.logger.info("user.register.request", { email: input.email });

    const normalizedEmail = input.email.trim().toLowerCase();
    const identity = await this.auth.createIdentity(normalizedEmail, input.password);
    const user = await this.repo.save({ email: normalizedEmail, identityId: identity.id });

    await this.mailer.sendWelcome(user.email);

    this.logger.info("user.register.success", { userId: user.id });
    return toDto(user);
  }
}
```

Do **not** extract auth, repo, mailer, and logger all at once.

First extract only the pure normalization and validation:

```ts
export function normalizeRegistration(input: RegisterInput): RegisterInput {
  if (!input.email.includes("@")) throw new Error("invalid_email");
  return { ...input, email: input.email.trim().toLowerCase() };
}
```

Then replace only that part in `UserService`.

This keeps side effects in place while proving the simplest extraction first.

---

## 5. Add characterization tests before risky movement

For legacy code, tests often describe intended behavior poorly. Before extraction, use Codex to add characterization tests around current behavior.

Ask Codex to capture:

- return values,
- exception types,
- exception messages,
- log event names,
- log payload fields,
- metrics,
- feature flag behavior,
- ordering of side effects,
- retries and timeouts,
- public API shape.

### Test levels per extraction

| Test level | Purpose | Example |
|---|---|---|
| Characterization test | Public behavior stays the same | `register()` returns the same DTO and throws the same errors |
| Focused unit test | New extraction is independently tested | `normalizeRegistration()` trims, lowercases, and throws `invalid_email` |
| Side-effect contract test | Logs, metrics, calls, and order stay stable | Logger is called with identical event names and payload keys |
| API compatibility test | Public imports/exports still work | Old import path still resolves |
| Contract test | External consumers are not broken | Pact or API diff checks |

### Log preservation test

```ts
import { describe, it, expect, vi } from "vitest";

it("preserves log contract for register", async () => {
  const repo = { save: vi.fn().mockResolvedValue({ id: "u1", email: "a@b.com" }) };
  const auth = { createIdentity: vi.fn().mockResolvedValue({ id: "id1" }) };
  const mailer = { sendWelcome: vi.fn().mockResolvedValue(undefined) };
  const logger = { info: vi.fn() };

  const service = new UserService(repo as any, auth as any, mailer as any, logger as any);

  await service.register({ email: " A@B.com ", password: "pw" });

  expect(logger.info.mock.calls).toEqual([
    ["user.register.request", { email: " A@B.com " }],
    ["user.register.success", { userId: "u1" }],
  ]);
});
```

For looser but still useful structured checks:

```ts
it("preserves log schema", async () => {
  const logger = { info: vi.fn() };

  await service.register({ email: "a@b.com", password: "pw" });

  expect(logger.info).toHaveBeenNthCalledWith(
    1,
    "user.register.request",
    expect.objectContaining({ email: expect.any(String) }),
  );

  expect(logger.info).toHaveBeenNthCalledWith(
    2,
    "user.register.success",
    expect.objectContaining({ userId: expect.any(String) }),
  );
});
```

---

## 6. High-level restructuring: plan architecture before moving files

Treat high-level restructuring as a separate phase from file extraction.

Take 1 is right that you should not mix extraction and folder moves. Take 2 adds the stronger production pattern: introduce boundaries and compatibility layers first, then move files.

### No-edit architecture map prompt

```text
Analyze the current module/folder structure. Do not edit.

Produce:
- current top-level domains/modules,
- request flows,
- classes that belong together,
- misplaced classes,
- circular dependencies,
- public entry points,
- CLI commands,
- jobs,
- events,
- scheduled tasks,
- OpenAPI endpoints,
- suggested target folder structure,
- move-only milestones,
- compatibility shims needed for old imports/public APIs,
- validation command per milestone.
```

### Good folder/module grouping rules

Use these defaults:

- Group by **domain or capability** first.
- Use technical layers only where the repo already clearly follows that style.
- Keep public entry points stable.
- Put adapters near boundaries: HTTP, DB, queues, CLI, third-party APIs.
- Keep domain rules away from framework code.
- Keep extracted classes small and named by responsibility.
- Avoid vague `utils` folders.
- Prefer names that explain purpose: `billing/discounts`, `customer/identity`, `observability/logging`, not `helpers`.

### Example target structure

```text
apps/
  api/
  web/

modules/
  billing/
    api/
    application/
    domain/
    infrastructure/

  customer/
    api/
    application/
    domain/
    infrastructure/

shared/
  kernel/
  observability/
```

A simple rule:

```text
apps -> module api -> application -> domain
                         |
                         -> infrastructure
```

Domain should not depend on infrastructure. Application may coordinate domain and infrastructure. API adapts external callers to application services.

---

## 7. Use staged restructuring, not big-bang moves

For productive codebases, the safest restructuring sequence is:

1. **Inventory current behavior and public paths**
2. **Add facades or deprecated re-exports**
3. **Add boundary tests**
4. **Move files/classes**
5. **Rewrite imports mechanically**
6. **Run parity gates**
7. **Canary or staged rollout**
8. **Remove old paths only at an explicit cleanup milestone**

### Restructuring approaches

| Approach | Best for | Benefit | Risk |
|---|---|---|---|
| Move-only refactor | Internal code with no external consumers | Fast | Hidden import breaks |
| Facade + boundary tests | Most production monoliths and monorepos | Good safety/speed balance | Temporary extra layer |
| Strangler or compatibility layer | Public APIs, high-traffic paths, risky migrations | Best rollback story | More temporary complexity |
| Codemod-driven migration | Many repeated import changes | Reproducible and scalable | Bad codemod spreads mistakes fast |

The best default is:

> **Facade first, move second, remove later.**

### Move-only milestone prompt

```text
Implement Milestone 1 only: move <specific files/classes> into <target folder>.

Constraints:
- Move code with minimal edits.
- Update imports only as required.
- Keep old import paths working using re-exports, adapters, or facades.
- Do not change behavior, names, signatures, logs, errors, tests, dependencies, or formatting.
- Do not combine this with cleanup.
- Run targeted tests and typecheck.
```

### Staged rollout prompt

```text
Plan the restructuring of @src/billing into @modules/billing.

Use branch-by-abstraction:
- first facade or deprecated re-exports,
- then file movement,
- then import rewriting,
- then removal of old paths.

Deliver:
- milestones,
- rollback plan per milestone,
- required codemods,
- CI changes,
- architecture tests to enforce new boundaries,
- public API compatibility plan.

Do not edit code yet.
```

---

## 8. Enforce architecture with tools, not just prompts

Codex can create a good structure, but the repo needs checks that keep it intact.

The takes mention these categories:

| Purpose | TS/JS | JVM | Python |
|---|---|---|---|
| Architecture boundaries | `eslint-plugin-boundaries`, `dependency-cruiser` | ArchUnit | Import Linter |
| Mechanical moves/codemods | `jscodeshift`, `ts-morph` | OpenRewrite | LibCST |
| Cycles and dependency graphs | `dependency-cruiser` | ArchUnit | Import Linter |
| API compatibility | API Extractor, `tsd` | Revapi, japicmp | Contract/API tests |
| HTTP API compatibility | `oasdiff`, `openapi-diff` | `oasdiff`, `openapi-diff` | `oasdiff`, `openapi-diff` |
| Consumer contracts | Pact | Pact | Pact |

Practical rule:

> Add architecture checks before large file moves. Otherwise Codex can create a new structure faster than your toolchain can defend it.

### Example TypeScript import rewrite with `ts-morph`

```ts
import { Project } from "ts-morph";

const project = new Project({ tsConfigFilePath: "tsconfig.json" });

for (const sf of project.getSourceFiles("src/**/*.ts")) {
  for (const imp of sf.getImportDeclarations()) {
    const spec = imp.getModuleSpecifierValue();

    if (spec.startsWith("@legacy/billing")) {
      imp.setModuleSpecifier(spec.replace("@legacy/billing", "@modules/billing"));
    }
  }
}

await project.save();
```

Ask Codex to generate codemods only after it has produced a move plan and compatibility plan.

---

## 9. Preserve everything: behavior, logs, tests, APIs

“Nothing broke” is not one test. It is a layered proof system.

### Preservation checklist for every prompt

Ask Codex to preserve:

- log statements,
- log levels,
- log message strings,
- structured log fields,
- metrics,
- spans,
- analytics events,
- audit events,
- exception types,
- exception messages,
- error wrapping,
- retry behavior,
- public classes,
- public functions,
- constructors,
- exports,
- imports,
- routes,
- CLI flags,
- request/response shapes,
- serialized JSON,
- database assumptions,
- feature flags,
- config keys,
- environment variables,
- ordering,
- timing,
- caching,
- concurrency behavior,
- side effects,
- comments explaining business rules.

### Validation prompt after each pass

```text
Review your own diff.

Check specifically for:
- removed or changed logs,
- changed log levels,
- changed structured log fields,
- changed metrics or event names,
- changed public APIs/import paths,
- changed error messages/types,
- changed behavior or ordering,
- changed feature-flag behavior,
- unintentional formatting churn,
- missing tests,
- broken backward compatibility.

Then run:
- targeted tests for the changed area,
- typecheck/compile,
- lint if relevant,
- API compatibility checks if public surfaces changed,
- contract checks if service boundaries changed.

Report any failures and fix only failures caused by this refactor.
```

### Verification matrix

| Must not break | Gate |
|---|---|
| Public TS exports | API Extractor API report, `tsd` |
| Public JVM APIs | Revapi, japicmp |
| HTTP/OpenAPI contracts | `oasdiff`, `openapi-diff` |
| Consumer-provider contracts | Pact provider verification, `can-i-deploy` |
| Logs and metrics | Characterization tests with golden event names/payload keys |
| Feature flag keys | Focused assertions or config contract tests |
| Import boundaries | dependency-cruiser, ArchUnit, Import Linter |
| Public import paths | Re-export tests or compile checks against old paths |
| Behavior | Unit, integration, characterization, and regression tests |
| Rollback | Small milestone PRs, feature flags, facades |

### Example API/contract commands

```bash
pnpm api-extractor run --local --verbose
pnpm tsd
mvn -q revapi:check
docker run --rm -t oasdiff/oasdiff breaking old-openapi.yaml new-openapi.yaml
openapi-diff old-openapi.yaml new-openapi.yaml --fail-on-incompatible
pact-broker can-i-deploy --pacticipant billing-api --version $GIT_SHA --to-environment production
```

Use only the gates that fit your stack. The key idea is to make compatibility visible and automatic.

---

## 10. CI guardrails for refactor branches

Take 2 adds an important point: large refactors need CI lanes that separate architecture, unit, contract, and API checks.

Example:

```yaml
name: refactor-guardrails

on:
  pull_request:
  push:
    branches: [main]

jobs:
  architecture:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pnpm install
      - run: pnpm depcruise src --output-type err-long
      - run: pnpm eslint .

  verify:
    needs: architecture
    strategy:
      matrix:
        lane: [unit, contract, api]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pnpm install
      - run: pnpm run verify:${{ matrix.lane }}
      - uses: actions/upload-artifact@v4
        with:
          name: verify-${{ matrix.lane }}
          path: reports/
```

This gives Codex a real safety net. It also makes failures easier to classify.

---

## 11. Use Codex features intentionally

### `AGENTS.md`

Use for durable repo rules:

- conventions,
- test commands,
- build commands,
- review expectations,
- forbidden changes,
- compatibility rules.

### Skills

Use Skills for repeated workflows, such as:

- splitting large files,
- creating characterization tests,
- moving modules,
- writing codemods,
- checking API compatibility,
- reviewing logs/metrics preservation.

A Skill should define the method, not a one-off task.

### ExecPlans and `/goal`

Use these for long, multi-step refactors.

Good `/goal` shape:

```text
Goal: split the billing module into domain, application, api, and infrastructure folders without changing behavior.

Stop conditions:
- old public imports still work,
- API compatibility checks pass,
- contract tests pass,
- log event names and metric names are unchanged,
- no unrelated formatting churn,
- all milestone validation commands pass.

Work in milestones. Before each milestone, state the current behavior, planned change, compatibility plan, and validation command.
```

### `codex exec`

Use for scripted or CI-like Codex work:

```bash
codex exec --cd . --sandbox workspace-write "Map the Billing module and return extraction candidates with validation steps."
codex exec --cd . --sandbox workspace-write --json "Run only Milestone A and return the result as JSON."
codex exec resume --last "Continue from the failed contract tests and fix only regressions caused by this refactor."
```

### Subagents

Use subagents for parallel analysis, not uncontrolled patching.

Example:

```text
Spawn three subagents and wait for all results.

Agent A:
Find SRP hotspots and extraction candidates in @src/user.

Agent B:
Inventory affected tests, snapshots, and contract checks.

Agent C:
Inventory public APIs, logs, metrics, event names, and feature flags.

Combine the results into one prioritized extraction matrix.

Do not edit code.
```

This is useful for exploration. Keep actual code changes centralized and milestone-based.

---

## 12. Backwards compatibility rules

For public APIs, never treat a move as “just a move.”

Use this sequence:

1. Introduce the new path.
2. Keep the old path through a deprecated re-export, wrapper, or facade.
3. Add compatibility tests.
4. Move internal consumers gradually.
5. Use API/contract checks in CI.
6. Remove the old path only at an explicit removal milestone.

Example:

```ts
// old path: src/billing/index.ts
export {
  BillingService,
  createInvoice,
  calculateDiscount,
} from "@modules/billing/api";

// optional deprecation comment
/**
 * @deprecated Import from @modules/billing/api instead.
 */
```

For runtime paths, use branch-by-abstraction, feature flags, or a facade so rollback is visible.

Operational rule:

> No milestone without a fallback.

---

## 13. A complete recommended workflow

### Phase 0: Prepare

1. Commit or stash current work.
2. Add/refine `AGENTS.md`.
3. Add or update architecture docs.
4. Confirm test commands.
5. Add basic architecture/API/contract gates where missing.

### Phase 1: Map

Ask Codex to inspect, not edit.

```text
Analyze @src/billing. Do not edit.

Produce:
- responsibilities by file,
- public APIs,
- side effects,
- logs,
- metrics,
- feature flags,
- tests,
- dependency cycles,
- extraction candidates,
- target module structure,
- lowest-risk first milestone.
```

### Phase 2: Add missing characterization tests

Before risky movement:

```text
Add only the minimal characterization tests needed to capture current behavior for <public method/module>.

Capture:
- return values,
- exceptions,
- logs,
- metrics,
- feature flags,
- side-effect ordering.

Do not refactor production code.
```

### Phase 3: Extract one responsibility

Use the one-responsibility extraction prompt.

Commit after validation.

### Phase 4: Repeat extraction

Keep each pass small.

Good PR titles:

```text
Extract email normalization from UserService
Extract billing discount calculation into domain service
Move invoice DTO formatting into billing/api
Add facade for legacy billing imports
Move billing application services to modules/billing/application
```

Bad PR titles:

```text
Clean up billing
Refactor user module
Restructure services
Big architecture cleanup
```

### Phase 5: Restructure modules

Only after enough internal responsibilities are separated:

1. Add facades/re-exports.
2. Add boundary checks.
3. Move files.
4. Rewrite imports.
5. Run API/contract gates.
6. Keep old paths until cleanup milestone.

### Phase 6: Final cleanup

Only remove old paths after:

- all internal imports use the new structure,
- public compatibility window is complete,
- API/contract checks are green,
- no consumers depend on deprecated paths,
- rollback plan is no longer needed.

---

## 14. Best prompt bundle

### A. Analyze large file

```text
/plan

Analyze @path/to/LargeFile only. Do not edit.

Identify SRP violations by listing:
- responsibilities,
- functions/classes involved,
- public APIs,
- side effects,
- logs/metrics/tracing,
- IO/DB/network calls,
- feature flags,
- tests,
- missing tests,
- lowest-risk extraction candidates.

Rank candidates from safest to riskiest.

End with exactly one recommended first extraction step and its validation command.
```

### B. Extract one responsibility

```text
Extract only <responsibility> from @path/to/LargeFile into <target module>.

Rules:
- no behavior changes,
- no public API breakage,
- no removed or changed logs,
- no changed exception types/messages,
- no unrelated formatting,
- no new dependencies,
- keep old imports working,
- add only minimal tests needed to preserve current behavior.

Before editing, state:
1. current behavior,
2. exact code to move,
3. compatibility plan,
4. validation command.

After editing:
- run validation,
- summarize diff,
- list remaining risk.
```

### C. Map module structure

```text
Analyze the current folder/module structure. Do not edit.

Produce:
- current domains/modules,
- request flows,
- misplaced classes,
- circular dependencies,
- public entry points,
- target structure,
- move-only milestones,
- compatibility shims,
- validation command per milestone.
```

### D. Move one milestone

```text
Implement Milestone <N> only.

Move:
- <file/class list>

Target:
- <new folder/module>

Rules:
- move code with minimal edits,
- update imports only as required,
- keep old import paths working,
- preserve logs/errors/tests/public APIs,
- do not reformat unrelated files,
- run targeted tests and typecheck.
```

### E. Review parity

```text
Review the current diff against main.

Check for:
- removed or changed logs,
- changed metrics,
- changed public exports/imports,
- changed OpenAPI/HTTP behavior,
- changed exception messages/types,
- changed feature flags,
- changed side-effect ordering,
- unrelated formatting churn,
- missing tests,
- broken backward compatibility.

If evidence is missing, add only the minimal check needed.

Do not introduce behavior changes.
```

---

## 15. Final checklist

Before accepting any Codex refactor, verify:

```text
[ ] The change has one clear purpose.
[ ] No dependency upgrades were mixed in.
[ ] No framework migration was mixed in.
[ ] No broad formatting rewrite was mixed in.
[ ] Public APIs still work.
[ ] Old import paths still work or have planned deprecation.
[ ] Logs are unchanged unless explicitly approved.
[ ] Metrics and event names are unchanged.
[ ] Exceptions and messages are unchanged.
[ ] Feature flags and config keys are unchanged.
[ ] Tests were added or updated only to preserve existing behavior.
[ ] Targeted tests pass.
[ ] Typecheck/compile passes.
[ ] Lint passes where relevant.
[ ] API/contract checks pass where relevant.
[ ] Diff has no unrelated churn.
[ ] Rollback is obvious.
[ ] The commit is small enough to review.
```

The safest operating principle is simple:

> **Make Codex prove every small step before you let it take the next one.**

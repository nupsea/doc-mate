# Core Beliefs

Operating principles for agent-assisted development of Doc-Mate.

## 1. Repository as System of Record

All project knowledge must live in the repository. Decisions made in Slack, email, or verbal discussions must be captured in `docs/` before they are actionable. If an agent cannot discover a constraint from the repo, that constraint does not exist.

## 2. Progressive Disclosure over Monolithic Instructions

Agents start with a short, stable entry point (AGENTS_GUIDE.md ~100 lines) that points to deeper sources of truth. No single file should try to encode everything.

## 3. Enforce Boundaries, Allow Autonomy Locally

Strict rules at layer boundaries (dependency directions, data validation, naming conventions). Within a layer, agents have freedom in how solutions are expressed.

## 4. Corrections Are Cheap, Waiting Is Expensive

In a high-throughput environment, fast iteration with follow-up fixes often beats blocking on perfection. This applies to PR merges, test flakes, and style nits -- not to security or data integrity.

## 5. Prefer Boring Technology

Dependencies and abstractions that are composable, API-stable, and well-represented in training data are easier for agents to model. When choosing between "clever" and "well-understood," choose well-understood.

## 6. Validate at Boundaries, Trust Internals

Parse and validate data at system boundaries (user input, external APIs, database reads). Internal function calls between trusted modules do not need redundant validation.

## 7. Capture Taste Once, Enforce Continuously

Human stylistic preferences and architectural decisions should be encoded into linters, structural tests, or documentation -- not repeated in every review comment.

## 8. Technical Debt Is a High-Interest Loan

Pay it down continuously in small increments. A recurring cleanup cadence catches bad patterns on a daily basis rather than letting them compound.

## 9. Legibility for Future Runs

Code does not need to match human aesthetic preferences. It must be correct, maintainable, and legible to future agent runs. Consistency across the codebase matters more than any individual style preference.

## 10. Design for the Constraint

When something fails, the fix is almost never "try harder." Identify what capability is missing -- tools, guardrails, documentation -- and make it both legible and enforceable.

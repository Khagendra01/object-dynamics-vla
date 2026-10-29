# Project workflow

The repository owner requested that completed project changes be committed and
pushed to the current tracking branch as part of the work. Before committing,
review the status, staged diff, and recent history; stage only the changes
intended for the current task. Run appropriate checks and report their results.
Do not create empty commits, amend, force-push, or overwrite unrelated work.
If pushing is blocked or there is no tracking branch, report the problem rather
than changing repository configuration without asking.
For public commits, use the owner's GitHub noreply email if the configured local
email is private; do not change global Git configuration.

Keep third-party datasets, pretrained model checkpoints, credentials, local
environment files, and raw experiment outputs out of the public repository.
Document real experiments with their provenance and limitations, and distinguish
oracle simulator inputs from camera-only perception. Do not treat held-out test
initializations as untouched after they have been used.

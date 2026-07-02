# Stream Runtime Surface

- Generated at: `2026-07-02T20:44:49.427+00:00`
- Sample size: `500`
- Synthetic rows: `500`
- Public music songs: `100`

## Hero Signals

- **Representation parity**: `0.732` — Share balance in the active representation surface.
- **Representation breadth**: `68.6%` — Synthetic lane diversity index, computed in Python and exported for the UI.
- **Attention concentration**: `46.5%` — Top-3 channel control share in the public music lane.
- **Notation-linked coverage**: `0.0%` — Share of catalog songs with directly linked score or notation support.

## Structural Bias Read

- **Band**: `watch`
- **Score**: `0.3644`
- **Summary**: This surface reads bias as movement: who stays central, which corridors narrow, and whether the public lane is covered well enough to support a claim.

## Runtime Notes

- The browser surface now has a generated backend state payload.
- SQLite persistence can store the latest runtime snapshot for inspection.
- Synthetic and public music lanes stay distinct all the way through export.

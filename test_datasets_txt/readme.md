
| Field | Meaning | Example |
|-------|---------|---------|
| `image_path` | Relative path to the test image. | `work7_rl/test/copymove/coverage/ture/97.tif` |
| `mask_path_or_None` | Path to the binary mask (`.png`) **for tampered images**; use the literal string `None` for **authentic (unaltered) images**. | `None` &nbsp;•&nbsp; `work7_rl/test/copymove/caisav1/masks/Sp_S_CND_A_pla0016_pla0016_0196_casia1_com.png` |

#### Example lines

```text
work7_rl/test/copymove/coverage/ture/97.tif None          # real
work7_rl/test/copymove/coverage/ture/98.tif None          # real
work7_rl/test/copymove/coverage/ture/99.tif None          # real
work7_rl/test/copymove/caisav1/tamper/Sp_S_CND_A_pla0016_pla0016_0196_casia1_com.png work7_rl/test/copymove/caisav1/masks/Sp_S_CND_A_pla0016_pla0016_0196_casia1_com.png  # tampered

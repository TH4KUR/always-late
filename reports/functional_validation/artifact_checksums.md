# Item-level result files of the functional study on Ada: sizes and sha256

Written 2026-10-23 01:17 IST from `sha256sum` and `stat` on the cluster (Q21). These files are NOT in git (large artifacts) and not on Hugging Face; they are the inputs of every analysis under `reports/functional_validation/results/`. Paths are relative to `/home2/karma/duan-verify/out/phase1/` on Ada. The jobs `func-eval-d` (floor) and `func-rescue-a` (rescue) were still running when this list was made; their files are added when they have ended. 50 files, 48.6 MB. A fetched copy can be checked with `sha256sum`.

| path | bytes | sha256 |
|---|---|---|
| `func-eval-a/func_lm_eval_FP.pt` | 2348759 | `6b422c53152906ac42e0eeef8853e489dd473ee77bfb21ded96f24229a928a73` |
| `func-eval-a/func_lm_eval_INT6.pt` | 2350491 | `6274bd8bc2bc871730bed56fff92230811a1324f40416cdaf5f6eab5e8865841` |
| `func-eval-a/meta_lm_eval.json` | 1994 | `45e8f464caf5036989024edfecce038909d7b411e7707206297efea50d898c52` |
| `func-eval-b/func_lm_eval_INT4.pt` | 2350491 | `04506dc3b5d1b2627dd50cab31fb0ae8509e940890a2ef13ae61985e236ddad5` |
| `func-eval-b/func_lm_eval_INT5.pt` | 2350491 | `939de073f5598f9085f44e5fbd881a37c578459bcb2b6aefb335e015812d57a8` |
| `func-eval-b/meta_lm_eval.json` | 1990 | `6a4643b0271eb5c610e62a940fc05957959e064fced749fda5411dcab825c491` |
| `func-eval-c/func_lm_eval_INT7.pt` | 2350491 | `a380a83850f7b433fd1ad5a7f3db2ca76484f5077a2ed6aa2b5cdcb800ef4f4f` |
| `func-eval-c/func_lm_eval_INT8.pt` | 2350491 | `ef9bebe45ffbc3f74617ed6700e248b8ffb2e386b50e504d5fc993545397d2b0` |
| `func-eval-c/meta_lm_eval.json` | 1993 | `2a84f7db9400598d0505938d5493079b130f62046816e95deb09a708f71f4e28` |
| `func-f4-INT4/func_lm_eval_FP.pt` | 589979 | `68065348e528ff5ff1f0f4d44b9f3f483e2170faa16c5f536f610d3f6828091e` |
| `func-f4-INT4/func_lm_eval_INT4.pt` | 590411 | `fc37180045ff3cf5179e14da9eb8707837f31e9134f31cf3aca344d14e080867` |
| `func-f4-INT4/meta_lm_eval.json` | 2423 | `aff61d477c777a93991b112a3881af318484dd5a0b84ea530850a1689a767416` |
| `func-f4-INT5/func_lm_eval_FP.pt` | 599219 | `4a1b93a318fc8ef2cab41447ae9a5dad822f8b85d98e788d18dda01cf3291ebd` |
| `func-f4-INT5/func_lm_eval_INT5.pt` | 599659 | `2430fcf34942fb366eb8afa8d147d7e7c1b5f7cd6d8b54a109ebba5b1444e2ed` |
| `func-f4-INT5/meta_lm_eval.json` | 2368 | `ea8731aa2eb6d9fc86779ac9baf4ac144c5192f7c41ddc88caa5cbe14e8b2704` |
| `func-f4-INT6/func_lm_eval_FP.pt` | 464467 | `fb5eca4cb3414eeac2c15a8af641ec9cf89d6fa6ddd46b829c41ab25baf2e71b` |
| `func-f4-INT6/func_lm_eval_INT6.pt` | 464811 | `a843804e480f8366d34a0ff52f343b40e7db042fb6ef6cf71e528b32d0b4a14d` |
| `func-f4-INT6/meta_lm_eval.json` | 2199 | `18ac9b70bae04bfa5133dad3d6806af6a1b954284325f44bf4c80e0d7f403401` |
| `func-mc-cal/func_mc_cal_cloze_FP.pt` | 503885 | `3452bb84f32a01ebc3be0b1bb20305af489068b6fb5216ff34ac1be18bcce11e` |
| `func-mc-cal/func_mc_cal_letter_FP.pt` | 817056 | `f21e144833c1c5ff911c49803987fac76257504016ed1dfc380967047aeda025` |
| `func-mc-cal/meta_mc_cal.json` | 1918 | `6673a66afd97fa002647c1219c95aef071dc57585792793f14f9189bd5bd001c` |
| `func-mc-eval-a/func_mc_eval_letter_FP.pt` | 2223231 | `b7cc98843e3ad37ab7a445b62d05a4f2cc98dddccc747ea3bfd9c1e50f3e1e24` |
| `func-mc-eval-a/func_mc_eval_letter_INT7.pt` | 2223495 | `86d6a621d1207ecf7f5d950b19e18202fe5be42fb7077dbb230c5c5e6a431203` |
| `func-mc-eval-a/func_mc_eval_letter_INT8.pt` | 2223495 | `aedd1b7836a2b795a79304e516f6fcf56b571c3f08f311c8bcf59938a7622afa` |
| `func-mc-eval-a/meta_mc_eval.json` | 2121 | `23f31d37ab435e015f773c3649d66abd80210d9b50e25ba72a54308bbe26dea3` |
| `func-mc-eval-b/func_mc_eval_letter_INT4.pt` | 2223495 | `347fde5161664640750ba2ffa346f8b955912f31a85d3366b75b71d57f60dc3c` |
| `func-mc-eval-b/func_mc_eval_letter_INT5.pt` | 2223495 | `1a9e019cc1229ce15b939882ee485806a7afb5609e2c1ac679e56edff276d350` |
| `func-mc-eval-b/func_mc_eval_letter_INT6.pt` | 2223495 | `d6dd96c274a3f9c04ce25fd276368b6a1bba609252d2ba38e39fe0ae1d4530ff` |
| `func-mc-eval-b/meta_mc_eval.json` | 2121 | `db094c32cfe69c35e1dd3d8522b57e2b110c9a154ae35f45a33dacace4a65bee` |
| `func-mc-pilot/func_mc_cal_letter_FP.pt` | 722619 | `cdb4e2a39ee8dc5bf0271a9254935f81ead768994359202ead7199ad9083ec24` |
| `func-mc-pilot/func_mc_cal_letter_INT4.pt` | 722947 | `0c1f1cfb775305d4138759204dfc43184e1cf45d57954fff36e60353cee05ce5` |
| `func-mc-pilot/meta_mc_cal.json` | 2035 | `8a9c05f6dd2d55ea48d919ebb81456bab0344bc078afa1a5f8a846761673a793` |
| `func-pilot-a/func_lm_pilot_FP.pt` | 2089645 | `c653d0ca44b404f5d660108d38f9a97050672e129ecf14e910b5cad49b68be59` |
| `func-pilot-a/func_lm_pilot_INT6.pt` | 2092449 | `96788a3cf503a34cbc7a642f3d3c06d3efa3623506ea3f9327b2ed9953ff90d8` |
| `func-pilot-a/func_lm_pilot_INT7.pt` | 2092449 | `a3d7e88e9dabd473541ebf676933168e6f0b36b55ed41df7ced1d48aa541e991` |
| `func-pilot-a/func_lm_pilot_INT8.pt` | 2092449 | `5c942fe33862e03438aee8b3d096c941b4b27b664f593e1257dbd33811d0039d` |
| `func-pilot-b/func_lm_pilot_INT12.pt` | 2093851 | `70075d0f2f00a160709462c783565e0ffb7647ee5b077f8a5bfccd456b6ef4d0` |
| `func-pilot-b/func_lm_pilot_INT16.pt` | 2093851 | `4bc0e4e41feb0edabdfa42ca36e486738e35a15bb3cabd3f9f4cc01574501480` |
| `func-pilot-b/func_lm_pilot_INT4.pt` | 2092449 | `57a7d78e4dddbb01c831149d3eeb98279ddd50ce91dc9a2abc9699382b2ab7f4` |
| `func-smoke/func_lm_pilot_FP.pt` | 53561 | `57a3049567c81277f4b059dd4b503dc41def758be8cd62df41490c932cd2aabe` |
| `func-smoke/func_lm_pilot_INT6.pt` | 53733 | `9c37b6586c11c6baf82ec8f6cd211d0c53c96cf30482112dcb430aca4624989c` |
| `func-smoke/meta_lm_pilot.json` | 1592 | `1722e44bc769f8609756fb0f9cd8179a91a6621960297b0a348788d841bc68cf` |
| `func-smoke2/func_lm_pilot_FP.pt` | 61377 | `900f04b329181ec5b8e3bf8830b4994926ac79c622736d6fb42b5b82c6619395` |
| `func-smoke2/func_lm_pilot_INT6.pt` | 89143 | `3b2a413865d62122ff9e3122ffc98ce00b0328c9cc72a221800c4f0e7598879a` |
| `func-smoke2/meta_lm_pilot.json` | 1709 | `2994cc2c0fb5f8cc3cc5e3e77a7cd6fe3b02268e2a4ed457c8a5f095fce6c1a8` |
| `func-smoke-mc/func_mc_cal_cloze_FP.pt` | 42505 | `a2a4eab7c499ca5be866e26bb872d5bc2741b02b8f675a00e4869ec74b7a12c3` |
| `func-smoke-mc/func_mc_cal_cloze_INT6.pt` | 42701 | `5909eaeed98ab410099ec72959180be8977e570f95231717177e20749dc63f95` |
| `func-smoke-mc/func_mc_cal_letter_FP.pt` | 36097 | `e53433bfc09b66a2d22684b4498f6ea418a8b422dfc38be9a33c1d8b20a6af81` |
| `func-smoke-mc/func_mc_cal_letter_INT6.pt` | 36233 | `f8d03aaea411cc6b6cc3a9a49134a5d385cf7152c371313abc9004d1a0a04f08` |
| `func-smoke-mc/meta_mc_cal.json` | 1799 | `43906bc0694a551afe38f6f19ba7a519a619d54a311dc517f29f96f3c22cd866` |

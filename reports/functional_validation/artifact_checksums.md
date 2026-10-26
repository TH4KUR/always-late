# Item-level result files of the functional study on Ada: sizes and sha256

Written 2026-10-23 01:17 IST from `sha256sum` and `stat` on the cluster (Q21). These files are NOT in git (large artifacts) and not on Hugging Face; they are the inputs of every analysis under `reports/functional_validation/results/`. Paths are relative to `/home2/karma/duan-verify/out/phase1/` on Ada. The job `func-eval-d` (floor, 35314) ended 2026-10-23 02:56 and its files were added at 03:10; the job `func-rescue-a` (rescue, 35319) ended at 04:14 and its files were added at 04:16; the F5 jobs `f5-a` and `f5-b` follow. The F5 smoke job (`f5-smoke2`, 35399) files were added at 03:40; the F5 full job A (`f5-a`, 35424) files at 06:32 ; the sensitivity job `f5-m8` at 07:13 ; job B (`f5-b`, 35435) at 07:57 ; `f5-m32` at 08:52 ; the K = all Belebele job `func-mc-ksens` at 12:33; the fixed-amplitude job `f5-fa` at 11:37; the positive-control runs `func-mc-pc-all` and `func-mc-pc-1024` (D-069) at 06:35; the D-035 per-feature status file `association-full/analysis-eq/feature_status.csv` (input of A2, D-066; equal to the rerun's) at 19:20; the swap control at INT8, INT6 and INT5 `f5-dir2` at 15:53; its combination with the swap control `f5-fa-dir` at 13:15; the direction-swap control `f5-dir` at 09:52. 116 files, 125.8 MB. A fetched copy can be checked with `sha256sum`.

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
| `func-eval-d/func_lm_eval_INT0.pt` | 1181275 | `7a8ad50717be43c999756ebf756af817d61059eb9996c1c4a53a89bc26e52abb` |
| `func-eval-d/func_lm_eval_INT12.pt` | 1181709 | `618ae1723756adc19fe9e76ef1aa1ae27440dbad6722b79e74899558f17e75f2` |
| `func-eval-d/func_lm_eval_INT16.pt` | 1181709 | `baee04a94ee533b01480d6271bcd0e53f2af57b4ac87caa175c8c6f3d4b48f1d` |
| `func-eval-d/meta_lm_eval.json` | 2158 | `c1e5fcb1dcb857983eb1e29333aa7738103a347ee95d769756e910cf9a963c62` |
| `f5-a/f5_stage_a.pt` | 478783 | `b64d3c4d2a085e77a35ed925761eea34325b81e282b86f5cdba65415b0364c42` |
| `f5-a/f5_state_FP.pt` | 4599045 | `c5f826c62b7e9e7f75cfbbe371556e5180d1db3202019eb7c8d9caf1b94b55ad` |
| `f5-a/f5_state_INT16.pt` | 4614618 | `900abd577fccbdb568ba7dd42749dbb189d9a4b4310e606889d6c21c6cb9ee23` |
| `f5-a/f5_state_INT8.pt` | 4615059 | `2fd3e3e8eb3c47fa01f3d0b34cde19e82dc9b4bfa6b9bd1c3b784a732b8e9839` |
| `f5-a/meta_f5.json` | 1675 | `fa895be5a350071ce5f1228733957faecbc5f4adaee22f9991210114f5f09ccf` |
| `f5-b/f5_stage_a.pt` | 478783 | `b64d3c4d2a085e77a35ed925761eea34325b81e282b86f5cdba65415b0364c42` |
| `f5-b/f5_state_FP.pt` | 4599045 | `c5f826c62b7e9e7f75cfbbe371556e5180d1db3202019eb7c8d9caf1b94b55ad` |
| `f5-b/f5_state_INT4.pt` | 4613587 | `7b977ae0e7110ee630232a0081b39518136ed6661253d9666ef004e7739b07df` |
| `f5-b/f5_state_INT5.pt` | 4613651 | `7ffbb931dbc8d263548238da9e1c607001c4c8536df63769328b622ddee796ea` |
| `f5-b/f5_state_INT6.pt` | 4614547 | `5982c54ea474a83e1c249121087b847a1af8824a209a998fd8f406f7c33046a2` |
| `f5-b/meta_f5.json` | 1713 | `cd982282fa623f19eae4c648c72bc3675b6516b806cbd3278e9ccc7260bbc1e7` |
| `f5-fa/f5_stage_a.pt` | 288255 | `62a6681266e103f170be894ef8653bb60d39e33e2bedceb0a55dd92f6472ff31` |
| `f5-fa/f5_state_FP.pt` | 376773 | `7ce5b2c40f10ce639935d4d083813eb4ffa19463b72464e531ad03df6361f1a9` |
| `f5-fa/f5_state_INT16.pt` | 381018 | `cdefcf3f8b0fb77f01894e0814b093b4160a812ee3db462ad99d028800a91593` |
| `f5-fa/f5_state_INT4.pt` | 381011 | `783263de667b6816cb4334ff3e71655d7213501250e2442e20e0ea8070da6aec` |
| `f5-fa/meta_f5.json` | 1728 | `3f838e8ece5ec11bc31f48d96d23512c98a1a80d65002c862374dd1f334fd635` |
| `f5-fa-dir/f5_stage_a.pt` | 291775 | `30448aa8e0a0a31298dd040d11654ed5cd4f9dd0e9e48ee1da9756cfa375c762` |
| `f5-fa-dir/f5_state_FP.pt` | 376645 | `ec328672bb694cd6e44045ebcccaf1165eb54f3f78edc5733c3d0691bdcd8b9e` |
| `f5-fa-dir/f5_state_INT16.pt` | 381082 | `d214c5a51607ebf86877648d6b200bbf2a6ac11a93088dde146eca156fa89db1` |
| `f5-fa-dir/f5_state_INT4.pt` | 380947 | `e48d670db0ae769d7ef414defdda84aefd22a1f6ce8e97b65cf5f621031a41f1` |
| `f5-fa-dir/meta_f5.json` | 1726 | `4ff3580922ae5d695c4c220ae62ee8a6de7f85e2ce1d86b79edda22f25bc25a4` |
| `f5-dir2/f5_stage_a.pt` | 291775 | `8a7d05d6c4d3fdba29d152b4f437c95c6eacce5154a917073a7101a323899488` |
| `f5-dir2/f5_state_FP.pt` | 376645 | `ec328672bb694cd6e44045ebcccaf1165eb54f3f78edc5733c3d0691bdcd8b9e` |
| `f5-dir2/f5_state_INT16.pt` | 381018 | `461824e553b787bbd68e99e6b020605f1a722fcf6747f835f273c7dc0a20eebb` |
| `f5-dir2/f5_state_INT8.pt` | 381075 | `28eb0d6214849610e519756c9d4ce7c679f160342887a5ed39f42b9d5948f9f8` |
| `f5-dir2/f5_state_INT6.pt` | 380819 | `0b03b6965342db148e92fa9329c7799d57075ad46f99940427804c595038a673` |
| `f5-dir2/f5_state_INT5.pt` | 380691 | `ed9258a2bf4ff6201720cc3af7d92d0060244ff8f7978f0a5f564f068cc70521` |
| `f5-dir2/meta_f5.json` | 1805 | `501dc7582e216a12b96699391d36684edd9b55503352b4771ea8420d0318d1c2` |
| `association-full/analysis-eq/feature_status.csv` | 15975546 | `3ab979852f3007987bfd364d334d9199e0cb2c3064d57cc932e287e70aec6a71` |
| `func-mc-pc-all/func_mc_eval_letter_FP.pt` | 2223231 | `08e651a9e99fa39426d90ddb4b5b902ce17267346a9c20957d772b7b54fc8773` |
| `func-mc-pc-all/meta_mc_eval.json` | 2030 | `2e7f68da2227fa8b21be6f3cc008d40f4292618922f112197b980c0f460e90f2` |
| `func-mc-pc-1024/func_mc_eval_letter_FP.pt` | 2223231 | `481f2b49fe829006c63187dc918f6d3f78e0b243d1a63796740af1c2ddec50d3` |
| `func-mc-pc-1024/meta_mc_eval.json` | 2031 | `35e8492145a75fbe0fa208e76fc27a6db99f4951dd2d8d637ea697da41c1a717` |
| `f5-dir/f5_stage_a.pt` | 291711 | `23c1fe9df1b712d355c7b83c76ba6c348df3a7d78535bb5dc6e4a06a35217d9a` |
| `f5-dir/f5_state_FP.pt` | 376645 | `ec328672bb694cd6e44045ebcccaf1165eb54f3f78edc5733c3d0691bdcd8b9e` |
| `f5-dir/f5_state_INT16.pt` | 381018 | `461824e553b787bbd68e99e6b020605f1a722fcf6747f835f273c7dc0a20eebb` |
| `f5-dir/f5_state_INT4.pt` | 380691 | `05719086615892af920d8e43adc610b0d7ba8f83cd375568942b897c74be868a` |
| `f5-dir/meta_f5.json` | 1701 | `05d9aa035b7089591d20b83ab74bbfb5a654348fe1d31efc7c85e1235f2ae529` |
| `f5-m32/f5_stage_a.pt` | 343167 | `c2648f6443f65574b4613b81f5267e9a72fcf28722da7be99793587873938f3b` |
| `f5-m32/f5_state_FP.pt` | 376837 | `cbaa91ba95c933d56d5bb15be1f6a7b28e9e31728c9747d1d5f016bd8fedf82e` |
| `f5-m32/f5_state_INT16.pt` | 381082 | `bc3f21c510f6e585ad15a3f3d06e51948bfcc783c7bd0fc1fabc3456538f83d6` |
| `f5-m32/f5_state_INT4.pt` | 380883 | `dbd8c05324d5c5dbbefdd56c3f5336759e3e6ff745205b3de7c13856c5a58780` |
| `f5-m32/meta_f5.json` | 1675 | `2023ab8fbfd740ec40248c7b50929e8bb0278f3c188770ab92d86981886c8950` |
| `f5-m8/f5_stage_a.pt` | 260735 | `31e41a1f673d7443ef1cfed0ed988cf5df6d68d679ff9a5a70a161f93973ea54` |
| `f5-m8/f5_state_FP.pt` | 376709 | `c3ec39fc01f25878d4f363a1d1cc25b5ea1338db1ab794a897ccafe494d8bec6` |
| `f5-m8/f5_state_INT16.pt` | 381018 | `ab8ff2f6bce501c0e0f3be7ae76fd739c6148c35c7bd3c6cf832e2e4fa3a3493` |
| `f5-m8/f5_state_INT4.pt` | 380755 | `891bc03f1a89cf5c5e21f5476c404afab513702bf2d03eb7354a2c6d1aaa8f3e` |
| `f5-m8/meta_f5.json` | 1673 | `a108fe8c7ae388cf5a55d4979201a3484741d53fd3e7ba808c72460c702602c3` |
| `f5-smoke2/f5_stage_a.pt` | 49791 | `ab44a6f96fea9a1f41f4c3e8f94acf93deae5a3faddd7de7829dd37cbda2a8db` |
| `f5-smoke2/f5_state_FP.pt` | 65541 | `39b1aa1bbabe625f59d9f11415737fdd085516ea0ac5c6d46d7ba128380a1180` |
| `f5-smoke2/f5_state_FP_rebatched.pt` | 65675 | `f3b2aac283727a9ad2f8581aeb43c8cdc3f7c2322be989c6deda2f651bc2eab6` |
| `f5-smoke2/f5_state_INT16.pt` | 67226 | `abe080d25b0b7069e39313e716e6f9583d632e3f7fa2fc9b941d48ec26892db3` |
| `f5-smoke2/f5_state_INT4.pt` | 67027 | `93a8d39848d69b9030d939db9f06bedfadb5adb6430850178951caba8d996b57` |
| `f5-smoke2/f5_state_INT8.pt` | 67155 | `9fe3b93ccc3b7fc535332a7cfb9d89a7818262a3f6093477ee4fece28248c52b` |
| `f5-smoke2/meta_f5.json` | 1733 | `3be26f5cea4a5a76da324afcba40ec94cc5eac475f6a437d278efbbcc9d00e3a` |
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
| `func-mc-ksens/func_mc_eval_letter_FP.pt` | 2223231 | `836beaaba4626bf7b1a58608e11be3aa54f695daaba9380b6f00652cfa6fbf5b` |
| `func-mc-ksens/func_mc_eval_letter_INT6.pt` | 2223495 | `ebfa069fb9ad39d2d4c35f7630bf677992b757428046ce3470eddf200475042c` |
| `func-mc-ksens/meta_mc_eval.json` | 2154 | `0e517d5a955e0994e70ca6887ac7c001dbcf83d237f0bd41eeaf3a6a054741f9` |
| `func-pilot-b/func_lm_pilot_INT12.pt` | 2093851 | `70075d0f2f00a160709462c783565e0ffb7647ee5b077f8a5bfccd456b6ef4d0` |
| `func-pilot-b/func_lm_pilot_INT16.pt` | 2093851 | `4bc0e4e41feb0edabdfa42ca36e486738e35a15bb3cabd3f9f4cc01574501480` |
| `func-pilot-b/func_lm_pilot_INT4.pt` | 2092449 | `57a7d78e4dddbb01c831149d3eeb98279ddd50ce91dc9a2abc9699382b2ab7f4` |
| `func-rescue-a/func_lm_eval_INT4.pt` | 1922987 | `7d5492628fc3e44197198d65c27be80991bc99387bfb300b7ab969d20a2063c4` |
| `func-rescue-a/func_lm_eval_INT5.pt` | 1922987 | `7134dc2d5bf2c1119287478d2a0443728a9d12fe04470c94d017b647e05fccd5` |
| `func-rescue-a/func_lm_eval_INT6.pt` | 1922987 | `67d723eca25f3d76431d399e69794c02b03c1093d0756c2e91bbb2a9bdb22bd2` |
| `func-rescue-a/meta_lm_eval.json` | 2116 | `b721dfada4be5c6cd14405e4939356cb06242d3d23422e65546a10eef8ecc6a6` |
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

## Inputs of Study G (graded gain), added 2026-10-24

Fetched from Ada with one `scp` per file, sha256 equal on both sides. Not in git (95.7 MB and 5.9 MB). Used by `scripts/g0/phase1_gain_baseline.py` (output `gain_study/baseline/`).

| path (relative to `/home2/karma/duan-verify/out/phase1/`) | bytes | sha256 |
|---|---|---|
| `association-full/association/eq/profiles.pt` (D-035 per-state profiles, FLORES devtest) | 95695711 | `4ed15f779ca8d6d3892758322bda5f7b73a6be2c7fd52f135ca11397a9e12c83` |
| `p1-mask/eq/selectivity.pt` (Phase 1 dev selectivity of the cohort) | 5893126 | `2f0422c2c4c4fdeae9024b8afa9c0b68a2baca926f0df5a16aac838e5a0db3f6` |

## Study G calibration job, added 2026-10-24 09:12

Fetched from Ada one file at a time, sha256 equal on both sides (`/home2/karma/duan-verify/out/phase1/gain-cal-FP/`, job 36834). Not in git (3.5 MB item file). Input of `scripts/g0/phase1_gain_ratios.py` (output `gain_study/calibration/`).

| path | bytes | sha256 |
|---|---|---|
| `gain-cal-FP/func_lm_pilot_FP.pt` | 3550109 | `4693ac57c6108782f2df05660f2a8b3d6b212bb4f804a5b4254ec582ec4cc051` |
| `gain-cal-FP/meta_lm_pilot.json` | 2190 | `40f158c9c8c19acf49180c8dbcebc26f3068f02a7802c89519bf40e25422a21f` |

## Positive control, third dose, added 2026-10-24 09:54

Fetched from Ada one file at a time (`/home2/karma/duan-verify/out/phase1/func-mc-pc-256/`, job 36760), sha256 equal on both sides; the files of the other two doses fetched again for the reproduction check have the checksums recorded above.

| path | bytes | sha256 |
|---|---|---|
| `func-mc-pc-256/func_mc_eval_letter_FP.pt` | 2223231 | `6369a6ad170f009cca121432a636a992f23dba5dc2549844c5375dd957ba58e7` |
| `func-mc-pc-256/meta_mc_eval.json` | 2027 | `49a4f5e7d3cbec7f3e521af9cd73d14393cff35ce31a5a7c7df5019a1454a6d1` |

## Study G main jobs, added 2026-10-24 19:50

Fetched from Ada one file at a time, sha256 equal on both sides (`/home2/karma/duan-verify/out/phase1/gain-<S>/`). Not in git (5.6 MB item files). Inputs of `scripts/g0/phase1_gain_analysis.py`. `gain-FP` 36925 and `gain-4` 36926: COMPLETED 12:07 and 12:10 (2 h 55 min and 2 h 58 min, exit 0, snapshot `repo_7fba511`, 999 configurations per state = 37 x 27, gates passed, complete, nothing skipped, 300 sentences per language with the ids of the freeze).

| path | bytes | sha256 |
|---|---|---|
| `gain-FP/func_lm_eval_FP.pt` | 5590755 | `5f88f1526b728eee45621138c05f60afe7f3a14243ec968a4632fcd188960d96` |
| `gain-FP/meta_lm_eval.json` | 2028 | `d237279f8bb3bf4ecb6272cd364bad6c6820196700e424242fd27e925782802c` |
| `gain-4/func_lm_eval_INT4.pt` | 5594871 | `eb1c3a7c1c1ee97283085371b7d043d313b00996682d07c75f45a57b53e26a3e` |
| `gain-4/meta_lm_eval.json` | 2026 | `649f244bb68125a1520a5665b88185466fdbfc47d232cbb23d3ce7639dd43413` |

## Study G main jobs INT6 and INT5, added 2026-10-24 23:21

Fetched from Ada (slow link, one file at a time), sha256 equal on both sides. `gain-6` 37693 and `gain-5` 37694: COMPLETED 22:34 (2 h 53 min, exit 0, snapshot `repo_7fba511`, 999 configurations per state, gates passed, complete, nothing skipped, 300 sentences per language with the ids of the freeze: equal).

| path | bytes | sha256 |
|---|---|---|
| `gain-6/func_lm_eval_INT6.pt` | 5594871 | `c4c2dc55a1088413f4ceac8613cfd2c0d8870ba972d7573277ab91238e30a061` |
| `gain-6/meta_lm_eval.json` | 2025 | `896aa1c9a16733c3cbe4ebdbf6315d5099c645d2d40f63b75588103d5766380c` |
| `gain-5/func_lm_eval_INT5.pt` | 5594871 | `46abf84a36623e4a8d8c271848159f780b466d4c99993ad063deedde0a9d9d85` |
| `gain-5/meta_lm_eval.json` | 2025 | `bbfb17d30616b411cdb93ca1d852d9a2b6cc87752376bf64dbe99010743afd9b` |

## Study G: INT7 and INT8 main states, the extension C jobs, the INT4 calibration, added 2026-10-25 13:06

Fetched from Ada, sha256 equal on both sides (17 files compared, 0 differences). `gain-7` 37947 and `gain-8` 37948 (main states, 2 h 54 and 3 h 01, snapshot `repo_7fba511`, 999 configurations, 300 sentences with the ids of the freeze); `gainc-smoke` 38123 (mechanics check, 3 languages x 40 reserve sentences, FP and INT4); the extension C jobs on the 562 reserve sentences with five control draws and no zero-ablation (snapshot `repo_ab7790e`, 675 configurations each): `gainc-FP` 38125, `gainc-6` 38126, `gainc-5` 38127 and **`gainc-4` 38124, which ran on an RTX 4070 (node gnode068, labelled 2080ti by mistake) and is therefore a hardware sensitivity run only (D-079); the INT4 extension verdicts use `gainc-4b` 38720**; `gain-cal-INT4` 38128 (informational calibration, pilot split). Every other job of the study ran on an RTX 2080 Ti (checked in the logs).

| path | bytes | sha256 |
|---|---|---|
| `gain-7/func_lm_eval_INT7.pt` | 5594871 | `c72a8dbb9c6b4b823be0e69db4d895c0f9983561b4c4fa82368155f098191ac9` |
| `gain-7/meta_lm_eval.json` | 2027 | `16eb08930a51660b018c62620bef360ed35d44f62b3c45dfb43fd9ec286695fe` |
| `gain-8/func_lm_eval_INT8.pt` | 5594871 | `bf5b74f4d2e694e257b2116d8127ef6fcfbb38be972f29e9c030859aa10491b2` |
| `gain-8/meta_lm_eval.json` | 2027 | `1eccb4bae3a5a41124070de19735eb2d0c2285bf6d440d15c2e546a921ab6fc8` |
| `gainc-smoke/func_lm_eval_FP.pt` | 104023 | `956cad7d5faf107596e50cd5eeddfcd02993ba2319be52bd4c415dfc46daad12` |
| `gainc-smoke/func_lm_eval_INT4.pt` | 104347 | `c326ab6df5ba75cb20c4f3e65e273952119cf7ba686f7be5dec1df4cd2348e1d` |
| `gainc-smoke/meta_lm_eval.json` | 1758 | `1115af836a57258dce047dab6d7d500f76add40efe6266b54ff82b1f739b5d76` |
| `gainc-4/func_lm_eval_INT4.pt` | 6816367 | `1f2f866a09c1c798180e16727806e264f26c44a500cb05d395fbf9ac77b0d2c5` |
| `gainc-4/meta_lm_eval.json` | 1853 | `ca278ed9bbe9299153e7c58db90d22c47ac7753e2a082caf3018c3005f9c9635` |
| `gainc-FP/func_lm_eval_FP.pt` | 6813547 | `25647295e7c87fcc50940829e6abd4a21a73d8a950555fc24d409bbcf0f271bb` |
| `gainc-FP/meta_lm_eval.json` | 1855 | `39cfdbcc7fb6148666729058f2c4d36eaf51c4ce58ca53f1e1fd2105e5c6d4be` |
| `gainc-6/func_lm_eval_INT6.pt` | 6816367 | `93415b4ac23ade21dc3ba9da86ed961521887106f96dc9149ebc34dc7ef46138` |
| `gainc-6/meta_lm_eval.json` | 1853 | `b72eb4d7ab3ba36bb426e220a3c359eae1eebe26d2dfc8866ac031bde65291e2` |
| `gainc-5/func_lm_eval_INT5.pt` | 6816367 | `f085fd642a2a399bc48e741f05a534281cfa184f65fc7d59cacebbd08a9a8d18` |
| `gainc-5/meta_lm_eval.json` | 1854 | `09517431713303f77c2e3d82783cffc53db4806e70c601b6d06f4032902452a1` |
| `gain-cal-INT4/func_lm_pilot_INT4.pt` | 3556493 | `9cae84832850183a3f6185b6cdc2c3fe510b0dd1ee21197dd086fd82b5bcbf8d` |
| `gain-cal-INT4/meta_lm_pilot.json` | 2193 | `a7cb945c8eff8f8e0ccfd6103316b553898951713f23b211e3d8c25a041070ea` |

Fetched 2026-10-25 16:37 IST from Ada, sha256 equal on both sides (2 files, 0 differences): `gainc-4b` 38720 (extension C, INT4, re-run of `gainc-4` on a verified RTX 2080 Ti, D-079; gnode057, 3 h 33 min, snapshot `repo_ab7790e`, 675 configurations, 562 reserve sentences per language, five control draws, gates passed, complete, nothing skipped; analysed in D-082).

| file | bytes | sha256 |
|---|---|---|
| `gainc-4b/func_lm_eval_INT4.pt` | 6816367 | `4bc7251c0186bc86aee8b5a36914c59eab50e4f160b3b685a58a8b86403a8cff` |
| `gainc-4b/meta_lm_eval.json` | 1855 | `b03c0db55bb37181517caceb5b32620dd035dd2ad46765784852b49b556ef7e0` |

Fetched 2026-10-25 17:55 IST from Ada, sha256 equal on both sides (2 files, 0 differences): `gain-sd-4` 38816 (extension B, INT4-defined groups, RTX 2080 Ti on gnode061, 3 h 13 min, snapshot `repo_dad2760`, commit `dad2760`, 999 configurations, 300 evaluation sentences per language with ids equal to the freeze, three control draws, gates passed, complete, nothing skipped; the manifest it used is `gain-sd-manifests/groups_manifest_INT4.json`, sha256 `4e89ac3a...` equal to `state_defined/meta.json`, whose own source hash `aef62872...` is the `manifest_source_sha256` of the meta file below).

| file | bytes | sha256 |
|---|---|---|
| `gain-sd-4/func_lm_eval_INT4.pt` | 5594871 | `0aada50213ba48477501f7a27beaadec95f1c95286929fe33c7c9027203741ba` |
| `gain-sd-4/meta_lm_eval.json` | 2131 | `c070a217c92405bc1a7b48794df38db52b4aa961bcb27d2a4fab09cd56e1277b` |

Fetched 2026-10-25 19:52 IST from Ada, sha256 equal on both sides (2 files, 0 differences): `gain-sd-FP` 38817 (extension B, FP-defined state-view groups, which are also the FP reference group at FP; RTX 2080 Ti on gnode057, 2 h 51 min 51 s, snapshot `repo_dad2760`, commit `dad2760`, 999 configurations, 300 evaluation sentences per language with ids equal to the freeze, three control draws, gates passed, complete, nothing skipped; its manifest `gain-sd-manifests/groups_manifest_FP.json` has sha256 `c0bb1b58...` equal to `state_defined/meta.json`, whose own source hash `637480d4...` is the `manifest_source_sha256` of the meta file below).

| file | bytes | sha256 |
|---|---|---|
| `gain-sd-FP/func_lm_eval_FP.pt` | 5590755 | `49ddcfd9cc0f573293566deb8efbfb90ba0f94013f6fbaca6bfe7aa8618e7e07` |
| `gain-sd-FP/meta_lm_eval.json` | 2129 | `bdefe20e47f265a0344ea02102b6b0b30e8f75ef562e814b66ad64506ccf90e4` |

Fetched 2026-10-25 21:12 IST from Ada, sha256 equal on both sides (2 files, 0 differences): `gain-sd-6` 38818 (extension B, INT6-defined groups; RTX 2080 Ti on gnode061, 3 h 15 min 24 s, snapshot `repo_dad2760`, commit `dad2760`, 999 configurations, 300 evaluation sentences per language with ids equal to the freeze, three control draws, gates passed, complete, nothing skipped; its manifest `gain-sd-manifests/groups_manifest_INT6.json` has sha256 `08cc1c49...` equal to `state_defined/meta.json`, whose own source hash `e44c072b...` is the `manifest_source_sha256` of the meta file below).

| file | bytes | sha256 |
|---|---|---|
| `gain-sd-6/func_lm_eval_INT6.pt` | 5594871 | `919dfc1011dbf838bf2e357318d72dd6a3e84ec89926c83a0aa225dcd3003329` |
| `gain-sd-6/meta_lm_eval.json` | 2130 | `37901c9c16c44a8a1c179d7520541f4aa467b1c6278cc808c750bfa51ac05186` |

Fetched 2026-10-25 22:33 IST from Ada, sha256 equal on both sides (2 files, 0 differences): `gain-sd-FPref-4` 38869 (extension B, the FP reference groups run at INT4; RTX 2080 Ti on gnode048, 2 h 59 min 05 s, snapshot `repo_dad2760`, commit `dad2760`, 999 configurations, 300 evaluation sentences per language with ids equal to the freeze, three control draws, gates passed, complete, nothing skipped; it used `gain-sd-manifests/groups_manifest_FP.json` (sha256 `c0bb1b58...` equal to `state_defined/meta.json`; its source hash `637480d4...` is the `manifest_source_sha256` of the meta file below, the same as `gain-sd-FP`)).

| file | bytes | sha256 |
|---|---|---|
| `gain-sd-FPref-4/func_lm_eval_INT4.pt` | 5594871 | `a3af8c8b2c802a342e59d13510cf6f8d2c6eb9bd87fef5a868e824d544b13e1a` |
| `gain-sd-FPref-4/meta_lm_eval.json` | 2134 | `231194388b34990c08824b8490279ee65b67808b49bbdeb03529638c4d9df66c` |

Fetched 2026-10-26 00:32 IST from Ada, sha256 equal on both sides (2 files, 0 differences): `gain-sd-FPref-6` 38870 (extension B, the FP reference groups run at INT6; RTX 2080 Ti on gnode049, 2 h 48 min 56 s, snapshot `repo_dad2760`, commit `dad2760`, 999 configurations, 300 evaluation sentences per language with ids equal to the freeze, three control draws, gates passed, complete, nothing skipped, no CUDA warning in the log; it used `gain-sd-manifests/groups_manifest_FP.json` (sha256 `c0bb1b58...`, source hash `637480d4...`, the same as `gain-sd-FP` and `gain-sd-FPref-4`)). With it the five jobs of extension B are in.

| file | bytes | sha256 |
|---|---|---|
| `gain-sd-FPref-6/func_lm_eval_INT6.pt` | 5594871 | `d7d7b727a2a63f9eddcda17de12d9ef7f4d1961e6b885c506ea6765482573c0e` |
| `gain-sd-FPref-6/meta_lm_eval.json` | 2133 | `d2563cccadc7e5458e6b1198e61853b5d7593838c712a4de1e378b87248b996b` |

Fetched 2026-10-26 01:16 IST from Ada, sha256 equal on both sides (4 files, 0 differences): `promptcheck` 39406 (Task 17 part (b); RTX 2080 Ti on gnode049, 32 min 12 s, snapshot `repo_5ddf16e`, commit `5ddf16e`; calibration items, letter format, no edit, FP, 250 items per language for the 18 covered languages, item seed 0; gates passed, complete, nothing skipped, no CUDA warning). Variant `frozen` = the frozen letter prompt, variant `harness` = the lm-evaluation-harness prompt.

| file | bytes | sha256 |
|---|---|---|
| `promptcheck/frozen/func_mc_cal_letter_FP.pt` | 546493 | `1470272cf3d0adfcc80abb8f42e8d6e90b11198a91a00e1114d7f70c90888aff` |
| `promptcheck/frozen/meta_mc_cal.json` | 1965 | `ebb2e86686e49e2496b2c80ac7a0877dcaf2e1cc884006ac0032edd5cf8f7816` |
| `promptcheck/harness/func_mc_cal_letter_FP.pt` | 546493 | `dd11c4b93e1edfd44d5441fced09b3a7c244402a2723aa514a1447af7f675a36` |
| `promptcheck/harness/meta_mc_cal.json` | 1966 | `71e7294344f959ee1b92e3e7c1e452423d0c5042e478249470c6568f522038ea` |

Fetched 2026-10-26 07:11 IST from Ada, sha256 equal on both sides (the Llama-3.2-1B consistency check, D-087; RTX 2080 Ti on gnode048, no CUDA warning in the logs, snapshot `repo_0cf138e`, commit `0cf138e`): `llama-tag` 39717 (dev tagging pass, all 27 languages, equal token budget, selectivity file of the equal-budget stream; the manifest was built by the CPU job `llama-manifest` 39720) and `llama-lm-b` 39722 (states INT5 and INT4: sacct COMPLETED exit 0, 2 h 59 min 35 s, 999 configurations per state, 300 evaluation sentences per language with ids equal to the freeze, the Study G main preset, gates passed, complete, nothing skipped).

| file | bytes | sha256 |
|---|---|---|
| `llama-tag/eq/selectivity.pt` | 5144262 | `a2754c508d4b1380afb7a6780ca37d3d5c4be52cb5b064477103cef06e64c21a` |
| `llama-tag/eq/meta.json` | 5419 | `9942e316e98c0bf75337a8ff24b96ed90cdca6df236dabd229dbe1357b18cc0f` |
| `llama-tag/groups_manifest.json` | 116104 | `c1c735c210b458b189732ab3526c625948022a6ed4bde4dca31ae641d7ff2541` |
| `llama-lm-b/func_lm_eval_INT5.pt` | 5594871 | `af6b1ef56f121ce48fdb5d957c19e317130f76e0605fdeaadd3fa0db1b5862b5` |
| `llama-lm-b/func_lm_eval_INT4.pt` | 5594871 | `480058d6af2ec52afd23582357d6a24908db0009f4b9a0f77f61b79ced64294c` |
| `llama-lm-b/meta_lm_eval.json` | 2168 | `a307eb675a455cdafb1e4c21ba2777d9a18b38802039010d121ed4ba8f409c22` |

# Artifact checksums of the intervention-transfer study (D-088)

Written 2026-10-26 17:00 IST (Claude). The raw outputs are not in git (large binary chunk files). They are fetched from Ada with one `scp` per directory into `/home/karma/Documents/ANLP/project/transfer_runs/<job name>/` (a folder next to the repository) and compared with `sha256sum` against Ada's; every list below was compared and is equal on both sides. Each full job is appended here when it has been validated (the checklist is NEXT STEP item 1 of `reports/functional_validation/STATUS.md`). Nothing is deleted on Ada.

## Instrument jobs (correctness and throughput checks, not results; kept for the record)

### `transfer-instrument7` (Slurm 40360, feature level, INT16 and INT4, 12 features x 4 contexts, snapshot `repo_4828e01`)
```
83edb21f4e6de0a68148ab903dd8d29ae5602cd803922c28071634e9cd865e8d  gpu_mem.log
71c783184f31cc83810188fc685e2dc6a231c54fcbcc1d1d4bbb8eedda99ebf2  meta_transfer.json
afb811ac1a7a9ab78718b99b519560416cd7babcb2178ae2231021142b0c7606  transfer_INT16_c000.pt
a15ec57c70023a33cd97c743db089926a2813c31aaa32dcb4b9240ced1fc5a65  transfer_INT16_c001.pt
aaad08f746640978f0566f547d3b46d5e934e37b5e55015860c598ee9c07ae01  transfer_INT4_c000.pt
26a8935451a065e9cdb8e23cfb83b33fd3c448056bfdf68dca9892079073c8aa  transfer_INT4_c001.pt
```

### `transfer-group-smoke3` (Slurm 40361, group level, 3 languages x 20 sentences, INT16 and INT4, snapshot `repo_4828e01`)
```
bc4d0e86f9a9001a61bea600add35f65f82c66554cbecb50a86d58c200606d4f  gpu_mem.log
158d295c38286883b7be02ed940abc4bbe1da1c9c699f337747675234da2a6c9  group_INT16_c000.pt
9b4c10c431190323635d8cd00a0b5b6b345e9c2f9d8cb3a55885753fec9db265  group_INT16_c001.pt
aed4fea6488ae3df24596b712774d415d291d7ffd57a41b21437a1d551410d86  group_INT16_c002.pt
c0c04fe1557fa8047f58d0658e5e2f4d0275c4bf8bd2e5b70e602500f4d00f70  group_INT16_c003.pt
44607be625554eee3ac671f7710d0a40f8d52871089e97ddc869751852197e90  group_INT16_c004.pt
95c14a8b36b57d62f16b0cf4f9542ad9f5fbdcb6731afa3595f879b571fa2bb6  group_INT16_c005.pt
86450243ae01e7cd11d766fa8e9b16fc08a5667db24422937541ff002d12ccda  group_INT4_c000.pt
82c9cc0a44cae05aea1cc3e2c07fe73d4c58f4d1cc70023266a07ac5333dcd30  group_INT4_c001.pt
cc1b3975e2a8f82cde7192173238a8ab70f2ec51317fb1395a461fc0501eff77  group_INT4_c002.pt
80839037ba44d10f77e1f30ed0f11145c3fed5d97a24da9dfb98077e8554b407  group_INT4_c003.pt
d388d3f8dd664b3bda4ff5524d767c9f69399fa11d8d77088d4ebb781e1257ee  group_INT4_c004.pt
57637d3ff3ef1d93c38269e20d964bbf78b326b23842935262dbd1f619e4e7f2  group_INT4_c005.pt
0dd5e3fd82f86e725bc09090e0dcf0f3197298a17b307df4ca230ef5adba7b43  meta_transfer.json
```

### `transfer-ext-core2` (Slurm 40528, the external core, snapshot `repo_6d84d5b`; `core_external.json` and `core_external_record.json` are committed under `reports/intervention_transfer/`)
```
9b7a1a1bf16cb3e3f375dc2c2b8fe8204764a5d7fa69412167868a5b6e11fab5  core_external.json
08fcf310c2290464fff586726cc12b498ac052c0e662291213b4f929b07abb39  core_external_record.json
4a54edad75075c63bc8a0e454ade5893dfd2e220c98834a8ad07479363dd4af1  gpu_mem.log
366587c19d3d0313b69eba81f950188fbc37b5cd88978f7d77574ab32703cf45  max_activations_external.pt
e84b4b2dad703b8f5cb39fbbf742afc9d98743f0a2d143e6231ddaaadd92e6ce  meta_transfer.json
```

### `transfer-ext-group-smoke2` (Slurm 40529, external group level, 2 languages x 20 passages, INT16 and INT4, snapshot `repo_6d84d5b`)
```
e9cbe0f6f911ba98b254ebac0229faf15aaeca0827a618cd651c5c0b244e2964  gpu_mem.log
ed0d6e42caaead0759db53f5902eb217419154e620977530a5c08f60ddd1b60f  group_INT16_c000.pt
b92f7ec1d9bcf6dbe784f5550955282d63ba229b5782a5a47c76487dc83cc2e9  group_INT16_c001.pt
78169f7691c5fbd80b3585bbb77ab9d59406984b4b4a30069f7ca686ca9aa53a  group_INT16_c002.pt
e2a72fac97d2aaaa21ed15c5de01e946ab2085f01d1fec5db01278b9065a09f8  group_INT16_c003.pt
194e1c90ff887ff0a8279994375210f88d5c36de651816ddc7043180598de04a  group_INT4_c000.pt
0905f4a2315ef72675cf9fa70cc833e65400dca1fc35899ad1d91de39da59bac  group_INT4_c001.pt
50199c441daedadb3c35aea8bf35623e030f8726ecabf87dab0011443045d926  group_INT4_c002.pt
b5836dee6dbf88a231a6abc801b1f0bdebb390b6c68b51bcf624c4a7a2979b26  group_INT4_c003.pt
a0cafa253daab4c14c5bc0d4b33d6e7fb5a93ea88d6167ba2533a4eba196fe2c  meta_transfer.json
```

## Inputs of the secondary survival analysis

### `phase1-p1-mask-all` (Phase 1 survival output of the mask regime, all-languages pool, Ada `out/phase1/p1-mask/all/`, written 2026-10-18; used with `--survival` of `phase1_transfer_report.py`)
```
b453831d9bca49c0674e115439865aff05e0856e06d989bc313be98f47cb8acc  meta.json
a03961cbec9b36580493e6ec6750373e1cb76ad07409ccab31c5f99b06a8804d  survival.pt
```

## Full jobs

None validated yet at the time of writing.

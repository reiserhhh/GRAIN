# HM288: read-only zero-bias export recovery

2026-10-14 JST. Full primary capture384families/19200inquiries completed once,
1876.022882modelseconds<2400cap. Primary analysis then failed BEFORE writing
any analyzed outputs:head weights have3rows but exported zero bias has2entries.
Primary verify consequently failed for missing analyzed_seal.json. Preserve
both failures,all primary arrays,code/config/source/native endpoints and seals.

Cause:sealed HM280 collector exports np.zeros(2) when model.lm_head.bias isNone.
The new alphabet has3classes. The installed Qwen2 architecture constructs its
lm_head withbias=False;local weight metadata has no lm_head.bias and uses tied
input/output embeddings. The native outputs used the REAL head and saved all
three logits. This is a shape/export error,not failed scientific inference.

The companion may only replace the representation of the absent bias by an
equivalent3entry zero vector in its OWN directory. Require oldbias exactlyzero,
oldweightshape(3,1536),modelconfigQwen2ForCausalLM/tiedembeddings,architectural
bias=False declaration and checkpoint metadata absence before analysis. Bind
these evidence/code/input hashes. No estimating bias or offset from answers.

Reuse original batches/source/units/bank/cases/cacheguard/runtime by read-only
symlinks. Create only the companion head.npz and recovery provenance. Then use
the original frozen analyze/verify functions on this equivalent exported head.
Every derived actual logit is checked at the original.001adequate tolerance;
not a tighter accuracy search. No new model load,forward,training/API/private
material or source/question/role change. No primary file overwritten.

Prepare,analyze,verify,check must occur once in the separate
person_bound_observation_tree_r1/bias_export_recovery_r1/. Analysis/audit CPU
cap180s;not renewed. Any failure preserved,not automatic correction/retry.
Original30tests/4formatpass/testI001warning retained;do not edit sealed tests.
No experimentalsuccess claim from this technical correction. Science still
requires actual source/direction/assessability result readback.Original18/100/44
OPEN/noGit/Claude/automation/lockbox/API/private/train/oldqueue.

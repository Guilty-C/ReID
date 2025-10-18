# PRP-ReID — 可执行实现计划（DRY_RUN）
## 目标
以最小链路复现"属性描述→文本/图像嵌入→检索→评测(mAP, Rank-1)"
## 数据
DATASET_ROOT=[TBD]; 结构遵循 Market-1501（仅作路径说明，不复制数据）
## 流程/接口
captioner.py→captions.json → embed_text.py→text_embeds.npy；embed_image.py→img_embeds.npy → retrieve_eval.py→metrics.json
## 指标与验收
输出 metrics.json 含 {"rank1":float,"mAP":float,"n_query":int,"n_gallery":int}（非空即通过）
## 风险
API Key 缺失/限流[TBD]；无GPU[TBD]；数据路径不一致[TBD]
# CS224n notes + slides 参考文献路线图

## 一条主线

1. 先用分布式假设理解“词的意义如何从上下文学来”。
2. 再用反向传播、自适应优化与正则化学会训练这些表示。
3. 把表示放进依存解析、RNN/LSTM、CNN 和树结构，理解语言的局部、长程和层次性。
4. 用 Seq2seq、Attention 和 Transformer 处理跨位置对齐，再过渡到预训练、生成、QA 与多模态。

## 1. 词、语义与词向量

- J. R. Firth (1957), *A synopsis of linguistic theory 1930–1955* — “一个词由它所处的同伴来认识”，分布式语义的起点。[出处：Winter 2023 Note 1]
- George A. Miller (1995), [WordNet: A Lexical Database for English](https://doi.org/10.1207/s15516709cog1802_1) — 词汇资源、同义词与上下位关系。[出处：Winter 2023 Note 1]
- Yoshua Bengio et al. (2003), [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) — 用密集向量和神经网络建模语言。[出处：2019 notes 01、5、9；2023 Note 1；Transformer draft]
- Ronan Collobert et al. (2011), [Natural Language Processing (Almost) from Scratch](https://www.jmlr.org/papers/v12/collobert11a.html) — 用一个共享的神经网络表示多个 NLP 任务。[出处：2019 note 01；2023 Note 1；Transformer draft]
- Tomas Mikolov et al. (2013), [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — skip-gram、负采样和 word2vec 的训练框架。[出处：2019 notes 01、02、5；2023 Note 1；Transformer draft]
- Tomas Mikolov et al. (2013), [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 层次 Softmax、负采样与短语向量扩展。[出处：2019 notes 01、02、5；2023 Note 1；Transformer draft]
- Xin Rong (2014), [word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 展开 word2vec 的梯度和参数更新。[出处：2019 note 01；2023 Note 1；Transformer draft]
- Jeffrey Pennington, Richard Socher, Christopher Manning (2014), [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — 从全局共现统计学习词向量。[出处：2019 note 02；2023 Note 1]
- Eric H. Huang et al. (2012), [Improving Word Representations via Global Context and Multiple Word Prototypes](https://aclanthology.org/N12-1015/) — 多原型词义和全局上下文。[出处：2019 note 02]
- J. Yin, Y. Shen (2018), [On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/2018/hash/0e625a0e8a5f5bd5f9a44fa7a1bdc1f1-Abstract.html) — 词向量维度的评估与选择。[出处：2019 note 02]
- K. Batsuren et al. (2022), [UniMorph 4.0: Universal Morphology](https://aclanthology.org/2022.lrec-1.22/) — 跨语言词法标注资源。[出处：Winter 2023 Note 1]
- Hal Daumé III, [A Course in Machine Learning](http://ciml.info/) — 课程页面推荐的 ML 基础教材。[出处：课程页 Reference Texts]

## 2. 反向传播、优化与训练

- David E. Rumelhart, Geoffrey Hinton, Ronald Williams (1986), [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — 反向传播的经典论文。[出处：2019 note 01；2023 Note 1；Transformer draft]
- David E. Rumelhart, Geoffrey Hinton, Ronald Williams (1985), *Learning Internal Representations by Error Propagation*, in *Parallel Distributed Processing*, Vol. 1, ch. 8 — Transformer draft 引用的早期书籍章节。[出处：2019 notes 01、03、9；2023 Note 1；Transformer draft]
- Xavier Glorot, Yoshua Bengio (2010), [Understanding the Difficulty of Training Deep Feedforward Neural Networks](https://proceedings.mlr.press/v9/glorot10a.html) — 初始化、激活方差和梯度传播。[出处：2019 note 03]
- Nitish Srivastava et al. (2014), [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.jmlr.org/papers/v15/srivastava14a.html) — 随机子网络训练及测试期近似集成。[出处：2019 note 03]
- Stefan Wager et al. (2013), [Dropout Training as Adaptive Regularization](https://proceedings.neurips.cc/paper/2013/hash/38db3aed920cf82ab059bfccbd02be6a-Abstract.html) — 从自适应正则化解释 dropout 的统计效果。[出处：2019 note 03]
- John Duchi, Elad Hazan, Yoram Singer (2011), [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://www.jmlr.org/papers/v12/duchi11a.html) — AdaGrad 按历史梯度缩放坐标学习率。[出处：2019 note 03；Transformer draft]
- Tijmen Tieleman, Geoffrey Hinton (2012), [RMSProp lecture](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — 用平方梯度移动平均避免 AdaGrad 学习率过快衰减。[出处：2019 note 03；Transformer draft]
- Diederik Kingma, Jimmy Ba (2015), [Adam](https://arxiv.org/abs/1412.6980) — 一阶矩、二阶矩和偏差修正的自适应优化器。[出处：2019 note 03；Transformer draft]
- Atilim Gunes Baydin et al. (2018), [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/abs/1502.05767) — 数值、符号和自动微分的关系。[出处：2019 note 03]

## 3. 依存句法与神经解析

- Joakim Nivre (2003), [An Efficient Algorithm for Projective Dependency Parsing](https://aclanthology.org/W03-3017/) — arc-standard/transition parsing 的起点。[出处：2019 note 04]
- James Henderson (2003), [Inducing History-Based Representations for Broad Coverage Statistical Parsing](https://aclanthology.org/W03-3007/) — 用神经方法表示解析历史。[出处：2019 note 09]
- Danqi Chen, Christopher Manning (2014), [A Fast and Accurate Dependency Parser using Neural Networks](https://aclanthology.org/D14-1082/) — 课程中使用的神经依存解析基线。[出处：2019 note 04]
- Sandra Kübler, Ryan McDonald, Joakim Nivre (2009), [Dependency Parsing](https://doi.org/10.2200/S00242ED1V01Y201005HLT010) — 统一的依存解析教材。[出处：2019 note 04]
- M. Andor et al. (2016), [Globally Normalized Transition-Based Neural Networks](https://arxiv.org/abs/1603.06042) — 局部迁移与全局正则化的对比。[出处：2019 note 04]
- Marie-Catherine de Marneffe et al. (2014), [Universal Stanford Dependencies](https://nlp.stanford.edu/pubs/dependencies-coling13.pdf) — 统一 Stanford 依存关系并支持跨语言比较。[出处：2019 note 04]
- [Universal Dependencies](https://universaldependencies.org/) — 与论文配套的跨语言树库和标注规范网站。[出处：2019 note 04]

## 4. 语言模型、RNN、GRU 与 LSTM

- Leonard E. Baum, Ted Petrie (1966), [Statistical Inference for Probabilistic Functions of Finite State Markov Chains](https://doi.org/10.1214/aoms/1177699147) — HMM 的统计基础。[出处：Transformer draft]
- Jeffrey L. Elman (1990), [Finding Structure in Time](https://doi.org/10.1016/0010-0277(90)90002-E) — 序列状态与简单 RNN。[出处：Transformer draft]
- Richard Socher et al. (2011), [Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions](https://aclanthology.org/P11-1015/) — 树结构与句子表示的早期实例。[出处：2019 note 09]
- Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013), [On the Difficulty of Training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) — 梯度消失、爆炸与裁剪。[出处：2019 note 05]
- Sepp Hochreiter, Jürgen Schmidhuber (1997), [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — 以 memory cell 和门控缓解长程梯度问题。[出处：2019 note 05]
- Kyunghyun Cho et al. (2014), [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 提出 GRU 和 RNN encoder–decoder。[出处：2019 note 05]
- Razvan Pascanu et al. (2013), [How to Construct Deep Recurrent Neural Networks](https://arxiv.org/abs/1312.6026) — 深层/双向 RNN 的结构设计。[出处：2019 note 05]
- Irsoy, Cardie (2014), [Deep Recursive Neural Networks for Compositionality in Text](https://aclanthology.org/P14-1105/) — 递归组合结构在文本建模中的应用。[出处：2019 note 05]
- David Silver et al. (2017), [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) — note 05 用于说明序列与长程延迟的背景对照。[出处：2019 note 05]

## 5. Seq2seq、Attention、子词与翻译

- Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014), [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 编码器–解码器范式。[出处：2019 note 06]
- Alex Graves (2012), [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711) — 早期 RNN 序列转导和语音识别框架。[出处：2019 note 06]
- Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014), [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 学习软对齐的 additive attention。[出处：2019 note 06；Transformer draft]
- Minh-Thang Luong et al. (2015), [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 比较 global/local、dot/general attention。[出处：2019 note 06；Transformer draft]
- Yonghui Wu et al. (2016), [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) — 大规模 NMT 工程和 GNMT 架构。[出处：2019 note 06]
- Melvin Johnson et al. (2017), [Google's Multilingual Neural Machine Translation System](https://arxiv.org/abs/1611.04558) — 一个模型覆盖多语言和 zero-shot 翻译。[出处：2019 note 06]
- Kyunghyun Cho et al. (2014), [On the Properties of Neural Machine Translation](https://arxiv.org/abs/1409.1259) — 序列条件概率与训练细节。[出处：2019 note 06]
- Kishore Papineni et al. (2002), [BLEU](https://aclanthology.org/P02-1040/) — 机器翻译的 n-gram 评估。[出处：2019 note 06]
- Zhixing Tan et al. (2018), [Learn to Encode and Translate for NMT](https://arxiv.org/abs/1808.09155) — 注意力对齐的可视化背景。[出处：2019 note 06]
- Zhaopeng Tu et al. (2016), [Modeling Coverage for Neural Machine Translation](https://arxiv.org/abs/1601.04811) — 记录已翻译内容，减少遗漏和重复。[出处：2019 note 06]
- Trevor Cohn et al. (2016), [Incorporating Structural Alignment Biases into an Attentive Neural Translation Model](https://aclanthology.org/Q16-1025/) — 将句法结构先验加入 attention。[出处：2019 note 06]
- Olivier Jean et al. (2015), [On Using Very Large Target Vocabulary for Neural Machine Translation](https://arxiv.org/abs/1412.2007) — sampled softmax 和 shortlist 处理大词表。[出处：2019 note 06]
- Caglar Gulcehre et al. (2016), [Pointing the Unknown Words](https://arxiv.org/abs/1503.01088) — 用 pointer 复制源句中的未知词。[出处：2019 note 06]
- Frederic Morin, Yoshua Bengio (2005), [Hierarchical Probabilistic Neural Network Language Model](https://proceedings.mlr.press/r5/morin06a.html) — 层次 softmax 降低大词表计算。[出处：2019 note 06]
- Rico Sennrich, Barry Haddow, Alexandra Birch (2016), [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 子词建模解决稀有词和开放词表。[出处：2019 note 06]
- Wang Ling et al. (2015), [Finding Function in Form](https://aclanthology.org/P15-2077/) — 字符级组合生成词和形态信息。[出处：2019 note 06]
- Minh-Thang Luong, Christopher Manning (2016), [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word–Character Models](https://aclanthology.org/N16-1105/) — word-level 与 character-level 混合翻译。[出处：2019 note 06]
- Kyunghyun Cho et al. (2015), [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1511.01432) — 将生成和情感控制结合的序列模型。[出处：2019 note 06]
- Dzmitry Bahdanau et al. (2017), [An Actor-Critic Algorithm for Sequence Prediction](https://arxiv.org/abs/1706.01905) — 用 actor–critic 直接优化序列级目标。[出处：2019 note 06]

## 6. QA、CNN 与递归组合

- Jason Weston et al. (2015), [Towards AI-Complete Question Answering](https://arxiv.org/abs/1502.05698) — 记忆网络和多跳问答的任务愿景。[出处：2019 note 07]
- Kumar et al. (2016), [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285) — 动态记忆和多跳注意的问答模型。[出处：2019 note 07]
- Kai Sheng Tai et al. (2015), [Improved Semantic Representations From Tree-Structured LSTM Networks](https://arxiv.org/abs/1503.00075) — TreeLSTM 将门控记忆扩展到句法树。[出处：2019 note 07]
- Zhiheng Huang et al. (2015), [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) — BiLSTM-CRF 的序列标注基线。[出处：2019 note 07]
- Caiming Xiong et al. (2017), [Dynamic Coattention Networks For Question Answering](https://arxiv.org/abs/1611.01604) — 课程在 QA 结尾指向的进阶阅读。[出处：2019 note 07]
- Nal Kalchbrenner et al. (2014), [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188) — 动态卷积和 k-max pooling。[出处：2019 note 08]
- Yoon Kim (2014), [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — 多通道句子分类 CNN。[出处：2019 note 08]
- Richard Socher et al. (2012), [Semantic Compositionality through Recursive Matrix-Vector Spaces](https://aclanthology.org/D12-1110/) — MV-RNN 以矩阵变换组合词向量。[出处：2019 note 09]
- Richard Socher et al. (2013), [Parsing with Compositional Vector Grammars](https://aclanthology.org/D13-1170/) — CVG 将句法解析与向量组合结合。[出处：2019 note 09]
- Richard Socher et al. (2013), [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) — RNTN 在树上建模组合情感。[出处：2019 note 09]
- John Lafferty, Andrew McCallum, Fernando Pereira (2001), [Conditional Random Fields](https://repository.upenn.edu/cis_papers/159/) — 序列标注的条件概率图模型。[出处：Transformer draft]
- Yann LeCun et al. (1989), [Backpropagation Applied to Handwritten Zip Code Recognition](https://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) — 卷积网络用于手写数字识别的早期示范。[出处：Transformer draft]

## 7. Self-Attention、Transformer 与大模型

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 多头自注意力、位置编码和 encoder–decoder。[出处：Transformer draft]
- Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016), [Layer Normalization](https://arxiv.org/abs/1607.06450) — 在单个样本内部归一化隐藏状态。[出处：Transformer draft]
- Jianlin Xu et al. (2019), [Understanding and Improving Layer Normalization](https://arxiv.org/abs/1911.07013) — 分析 LayerNorm 对表示和梯度的影响。[出处：Transformer draft]
- Ruibin Xiong et al. (2020), [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 解释 pre-LN/post-LN 的优化差异。[出处：Transformer draft]
- Ofir Press, Noah Smith, Mike Lewis (2022), [Train Short, Test Long](https://arxiv.org/abs/2108.12409) — ALiBi 位置偏置。[出处：Transformer draft]
- Jacob Devlin et al. (2019), [BERT](https://arxiv.org/abs/1810.04805) — masked language model 和 next-sentence prediction 的双向预训练。[出处：Transformer draft]
- Alec Radford et al. (2019), [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 的 decoder-only 预训练和 zero-shot 迁移。[出处：Transformer draft]
- Tom Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 的 in-context learning 和 scaling。[出处：Transformer draft]
- Colin Raffel et al. (2020), [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5 将任务统一成 text-to-text 接口。[出处：Transformer draft]
- BigScience Workshop et al. (2022), [BLOOM](https://arxiv.org/abs/2211.05100) — 开放多语言大模型的案例。[出处：Transformer draft]
- GPT-2、GPT-3 和 T5 的独立条目已列出；它们与 Transformer、LayerNorm 共同构成课程对现代预训练模型的最短历史线。

## 8. 多模态学习伴读

`Multimodal-Deep-Learning-CS224n-Kiela.pdf` 是仓库原有课件，它的引用和课程概念按“对齐表征→视觉语义→多模态 Transformer→数据集与实际体验”排序：

- Jason Weston et al. (2011), [WSABIE: Scaling Up To Large Vocabulary Image Annotation](https://www.ijcai.org/Proceedings/11/Papers/254.pdf) — 图像标签与文本空间的 scalable embedding。（出处：多模态课件）
- Andrea Frome et al. (2013), [DeViSE: A Deep Visual-Semantic Embedding Model](https://papers.nips.cc/paper_files/paper/2013/hash/7cce53cf90577442771720a370c3c723-Abstract.html) — 将视觉特征映射到语义词向量空间，实现 zero-shot 识别。（出处：多模态课件）
- Richard Socher et al. (2013), [Zero-Shot Learning Through Cross-Modal Transfer](https://arxiv.org/abs/1301.3666) — 用跨模态映射共享图像与文本表示。（出处：多模态课件）
- Elia Bruni et al. (2014), [Multimodal Distributional Semantics](https://aclanthology.org/J14-1005/) — 将视觉共现信息加入分布式词义。（出处：多模态课件）
- Douwe Kiela, Marco Baroni (2014), [Perceptual Models of Word Meaning](https://aclanthology.org/P14-1091/) — 用感知特征评估词义表征。（出处：多模态课件）
- Angeliki Lazaridou et al. (2015), [Multimodal Skip-Gram](https://aclanthology.org/Q15-1016/) — skip-gram 同时利用语言上下文和图像上下文。（出处：多模态课件）
- Ryan Kiros et al. (2014), [Unifying Visual-Semantic Embeddings](https://arxiv.org/abs/1411.2539) — 学习图像—句子共同嵌入用于检索。（出处：多模态课件）
- Faghri et al. (2018), [VSE++](https://arxiv.org/abs/1707.05612) — hard negative mining 改进视觉语义检索。（出处：多模态课件）
- Andrej Karpathy, Li Fei-Fei (2015), [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://arxiv.org/abs/1412.2306) — 对齐图像区域和句子片段生成描述。（出处：多模态课件）
- Oriol Vinyals et al. (2015), [Show and Tell](https://arxiv.org/abs/1411.4555) — CNN encoder + RNN decoder 的图像描述基线。（出处：多模态课件）
- Kelvin Xu et al. (2015), [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) — 用 soft attention 动态选择图像区域。（出处：多模态课件）
- Ian Goodfellow et al. (2014), [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — generator/discriminator 对抗训练生成样本。（出处：多模态课件）
- Scott Reed et al. (2016), [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396) — 以文本条件控制图像生成。（出处：多模态课件）
- Ross Girshick et al. (2014), [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](https://arxiv.org/abs/1311.2524) — R-CNN 将 region proposal 与 CNN 特征结合。（出处：多模态课件）
- Ross Girshick (2015), [Fast R-CNN](https://arxiv.org/abs/1504.08083) — 共享整图卷积特征，提升 R-CNN 训练/推理速度。（出处：多模态课件）
- Shaoqing Ren et al. (2015), [Faster R-CNN](https://arxiv.org/abs/1506.01497) — 用 RPN 端到端地产生候选区域。（出处：多模态课件）
- Joseph Redmon et al. (2016), [You Only Look Once](https://arxiv.org/abs/1506.02640) — 单阶段实时检测。（出处：多模态课件）
- Kaiming He et al. (2017), [Mask R-CNN](https://arxiv.org/abs/1703.06870) — 在检测框架上增加实例分割 mask 分支。（出处：多模态课件）
- Ali Sharif Razavian et al. (2014), [CNN Features off-the-shelf](https://arxiv.org/abs/1403.6382) — 固定 ImageNet 特征即可迁移到多种视觉任务。（出处：多模态课件）
- Alexey Dosovitskiy et al. (2021), [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — ViT 将图像 patch 作为 token 输入 Transformer。（出处：多模态课件）
- Ethan Perez et al. (2018), [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) — 用 feature-wise affine modulation 注入条件信息。（出处：多模态课件）
- Alec Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — CLIP 用图文对比学习获得 zero-shot 视觉能力。（出处：多模态课件）
- Chao Jia et al. (2021), [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) — ALIGN 以大规模噪声 alt-text 训练图文嵌入。（出处：多模态课件）
- Robin Rombach et al. (2022), [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — 在 latent space 扩散并用文本条件生成图像。（出处：多模态课件）
- Li et al. (2019), [VisualBERT](https://arxiv.org/abs/1908.03557) — 用 masked language modeling 预训练视觉—语言 Transformer。（出处：多模态课件）
- Lu et al. (2019), [ViLBERT](https://arxiv.org/abs/1908.02265) — 双流视觉/文本 Transformer 和 co-attention。（出处：多模态课件）
- Hao Tan, Mohit Bansal (2019), [LXMERT](https://arxiv.org/abs/1908.07490) — 视觉关系编码和跨模态 encoder。（出处：多模态课件）
- Douwe Kiela et al. (2019), [MMBT](https://arxiv.org/abs/1909.02950) — 将图像表示作为 multimodal transformer 的输入 token。（出处：多模态课件）
- Yunyang Huang et al. (2020), [Pixel-BERT](https://arxiv.org/abs/2004.00849) — 直接从图像像素和文本 token 学习跨模态表征。（出处：多模态课件）
- Yen-Chun Chen et al. (2020), [UNITER](https://arxiv.org/abs/1909.11740) — 统一视觉—语言预训练目标和跨模态对齐。（出处：多模态课件）
- Wonjae Kim et al. (2021), [ViLT](https://arxiv.org/abs/2102.03334) — 去掉 region detector 的轻量视觉语言 Transformer。（出处：多模态课件）
- Bugliarello et al. (2021), [Multimodal Pretraining Unmasked](https://arxiv.org/abs/2011.15124) — 综述多模态预训练目标和架构设计。（出处：多模态课件）
- Amanpreet Singh et al. (2022), [FLAVA](https://arxiv.org/abs/2112.04482) — 统一图像、文本和图文任务的多模态模型。（出处：多模态课件）
- Mingxing Wang et al. (2022), [SimVLM](https://arxiv.org/abs/2108.10904) — prefixLM 目标连接视觉编码和文本生成。（出处：多模态课件）
- Jiahui Yu et al. (2022), [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917) — 对比损失和 captioning 联合训练。（出处：多模态课件）
- Maria Tsimpoukelli et al. (2021), [Frozen](https://arxiv.org/abs/2106.13884) — 冻结语言模型、只训练视觉适配器实现少样本多模态推理。（出处：多模态课件）
- Jean-Baptiste Alayrac et al. (2022), [Flamingo](https://arxiv.org/abs/2204.14198) — 通过 gated cross-attention 处理交错图文上下文。（出处：多模态课件）
- Junnan Li et al. (2022), [BLIP](https://arxiv.org/abs/2201.12086) — 统一图文理解和生成并清洗 web 数据。（出处：多模态课件）
- Junnan Li et al. (2023), [BLIP-2](https://arxiv.org/abs/2301.12597) — 用 Q-Former 连接冻结视觉编码器和语言模型。（出处：多模态课件）
- Zhuoyi Zhang et al. (2023), [Multimodal Chain-of-Thought Reasoning](https://arxiv.org/abs/2302.00923) — 让视觉信息参与显式分步推理。（出处：多模态课件）
- Shaohan Huang et al. (2023), [KOSMOS-1](https://arxiv.org/abs/2302.14045) — 将视觉、语言和动作统一成通用多模态接口。（出处：多模态课件）
- Tsung-Yi Lin et al. (2014), [Microsoft COCO](https://arxiv.org/abs/1405.0312) — 图像、对象和场景描述的规模化数据集。（出处：多模态课件）
- Xinlei Chen et al. (2015), [Microsoft COCO Captions](https://arxiv.org/abs/1504.00325) — 图像描述生成和评测基准。（出处：多模态课件）
- Stanislaw Antol et al. (2015), [VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468) — 图像问答数据集和开放问题评测。（出处：多模态课件）
- Yash Goyal et al. (2017), [Making the V in VQA Matter](https://arxiv.org/abs/1612.00837) — VQA v2 通过平衡答案消除 language priors。（出处：多模态课件）
- Justin Johnson et al. (2017), [CLEVR](https://arxiv.org/abs/1612.06890) — 可组合视觉推理和程序化问题基准。（出处：多模态课件）
- Drew A. Hudson, Christopher D. Manning (2019), [GQA](https://arxiv.org/abs/1902.09506) — 结构化场景图上的组合视觉问答。（出处：多模态课件）
- Kiela et al. (2020), [The Hateful Memes Challenge](https://arxiv.org/abs/2005.04790) — 测试跨模态语境与隐含仇恨理解。（出处：多模态课件）
- Thrush et al. (2022), [Winoground](https://arxiv.org/abs/2204.03162) — 用图文组合关系而非词面匹配评测视觉语言模型。（出处：多模态课件）
- Alec Radford et al. (2022), [Whisper](https://arxiv.org/abs/2212.04356) — 大规模弱监督语音识别和翻译模型。（出处：多模态课件）
- Bapna et al. (2022), [mSLAM](https://arxiv.org/abs/2202.01374) — 多语言语音和文本的统一预训练。（出处：多模态课件）
- Rowan Zellers et al. (2021), [MERLOT](https://arxiv.org/abs/2104.08865) — 从视频、音频和文本学习时间对齐表示。（出处：多模态课件）
- Rowan Zellers et al. (2022), [MERLOT Reserve](https://arxiv.org/abs/2201.02639) — 扩大视频—语言预训练和叙事理解。（出处：多模态课件）
- Karl Moritz Hermann et al. (2017), [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551) — 在具身环境中联合学习语言和行动。（出处：多模态课件）
- Abhishek Das et al. (2018), [Embodied Question Answering](https://arxiv.org/abs/1711.11543) — 需要导航和视觉交互的问答任务。（出处：多模态课件）
- Alex Nichol et al. (2022), [Point-E](https://arxiv.org/abs/2212.08751) — 从文本生成三维点云。（出处：多模态课件）
- Douwe Kiela et al. (2015), [Grounding Semantics in Olfactory Perception](https://aclanthology.org/P15-2038/) — 将词义嵌入到嗅觉感知空间。（出处：多模态课件）
- Douwe Kiela, Stephen Clark (2017), [Learning Neural Audio Embeddings for Grounded Semantics in Auditory Perception](https://www.jair.org/index.php/jair/article/view/11122) — 用音频感知学习 grounded word embeddings。（出处：多模态课件）
- Patrick Lewis et al. (2020), [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 检索器和生成器结合，减少参数记忆压力。（出处：多模态课件）
- Simon Ott et al. (2022), [Mapping Global Dynamics of Benchmark Creation and Saturation in Artificial Intelligence](https://arxiv.org/abs/2203.04592) — 分析 benchmark 的创建、饱和与评测迁移。（出处：多模态课件）

## 9. 课程教材与工具

- Dan Jurafsky, James H. Martin, [Speech and Language Processing, 3rd ed. draft](https://web.stanford.edu/~jurafsky/slp3/)。
- Jacob Eisenstein, [Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)。
- Yoav Goldberg, [A Primer on Neural Network Models for Natural Language Processing](https://u.cs.biu.ac.il/~yogo/nnlp/)。
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, [Deep Learning](https://www.deeplearningbook.org/)。
- Delip Rao, Brian McMahan, [Natural Language Processing with PyTorch](https://library.stanford.edu/)。
- Lewis Tunstall, Leandro von Werra, Thomas Wolf, [Natural Language Processing with Transformers](https://transformersbook.com/)。
- Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)。
- Eugene Charniak, [Introduction to Deep Learning](https://mitpress.mit.edu/9780262045042/introduction-to-deep-learning/)。

## Slides 补充：解析、序列模型与生成评测

以下条目来自 Winter 2023 slides；前文 notes 已出现的论文不重复。每条单独列出，保留幻灯片出处。

- Warren S. McCulloch, Walter Pitts (1943), [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://doi.org/10.1007/BF02478259) — 阈值单元的早期形式化，连接神经元与逻辑计算。（来源：`slides/cs224n-2023-lecture03-neuralnets.pdf`）
- Mitchell P. Marcus et al. (1993), [Building a Large Annotated Corpus of English: The Penn Treebank](https://doi.org/10.1006/csla.1994.1014) — 现代句法和序列标注研究常用的树库基准。（来源：`slides/cs224n-2021-lecture04-dep-parsing-annotated.pdf`）
- Joakim Nivre, Jens Nilsson (2005), [Pseudo-Projective Dependency Parsing](https://aclanthology.org/W05-1506/) — MaltParser 体系中处理非投射依存的转换方法。（来源：`slides/cs224n-2021-lecture04-dep-parsing-annotated.pdf`）
- Timothy Dozat, Christopher Manning (2017), [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734) — 以双仿射打分替代局部迁移决策，成为图式神经解析基线。（来源：`slides/cs224n-2023-lecture04-dep-parsing.pdf`, `slides/cs224n-2021-lecture04-dep-parsing-annotated.pdf`）
- Paul J. Werbos (1988), [Backpropagation through time: what it does and how to do it](https://doi.org/10.1016/0893-6080(88)90007-8) — 给出循环网络跨时间反向传播的早期系统描述。（来源：`slides/cs224n-2023-lecture05-rnnlm.pdf`）
- Yoshua Bengio et al. (1994), [Learning Long-Term Dependencies with Gradient Descent is Difficult](https://doi.org/10.1109/72.279181) — 解释 RNN 中梯度消失/爆炸的根源。（来源：`slides/cs224n-2023-lecture06-fancy-rnn.pdf`）
- Felix A. Gers, Jürgen Schmidhuber, Fred Cummins (2000), [Learning to Forget: Continual Prediction with LSTM](https://doi.org/10.1162/089976600300015015) — 为 LSTM 引入 forget gate，是现代 LSTM 的关键补充。（来源：`slides/cs224n-2023-lecture06-fancy-rnn.pdf`）
- Kaiming He et al. (2016), [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — slides 用残差连接对照深层 RNN/CNN 的优化路径。（来源：`slides/cs224n-2023-lecture06-fancy-rnn.pdf`）
- Gao Huang et al. (2017), [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) — 密集跨层连接促进特征复用，作为深网络连接方式对照。（来源：`slides/cs224n-2023-lecture06-fancy-rnn.pdf`）
- Rupesh Kumar Srivastava et al. (2015), [Highway Networks](https://arxiv.org/abs/1505.00387) — 用门控高速通路缓解深层网络优化。（来源：`slides/cs224n-2023-lecture06-fancy-rnn.pdf`）
- Ari Holtzman et al. (2020), [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — 说明贪心/beam search 会放大重复和低多样性，提出 nucleus sampling。（来源：`slides/cs224n-2023-lecture10-nlg.pdf`）
- Angela Fan, Mike Lewis, Yann Dauphin (2018), [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) — 用层级规划生成长篇故事，展示长文本生成的结构性难题。（来源：`slides/cs224n-2023-lecture10-nlg.pdf`）
- Tim Meister et al. (2022), [Locally Typical Sampling](https://arxiv.org/abs/2202.00666) — 以局部熵约束采样，在质量和多样性间取平衡。（来源：`slides/cs224n-2023-lecture10-nlg.pdf`）
- Tianyi Zhang et al. (2020), [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) — 用上下文嵌入匹配候选和参考文本，超越 n-gram 重叠。（来源：`slides/cs224n-2023-lecture10-nlg.pdf`）
- Thibault Sellam, Dipanjan Das, Ankur Parikh (2020), [BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696) — 用预训练模型和人类评分学习生成质量回归器。（来源：`slides/cs224n-2023-lecture10-nlg.pdf`）
- Krishna Pillutla et al. (2021), [MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers](https://arxiv.org/abs/2111.00044) — 在分布层面评估生成文本与人类文本的差异。（来源：`slides/cs224n-2023-lecture10-nlg.pdf`）
- Abigail See, Peter J. Liu, Christopher D. Manning (2017), [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) — 复制源文本实体并生成摘要，缓解 OOV 和事实错误。（来源：`slides/cs224n-2023-lecture10-nlg.pdf`, `slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）

## Slides 补充：预训练、Prompting 与 RLHF

- Hector Levesque et al. (2012), [The Winograd Schema Challenge](https://doi.org/10.1007/978-3-642-33386-1_16) — 用需要常识消歧的代词题测试语言理解，而非表面词匹配。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Jason Wei et al. (2022), [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — 展示 few-shot 推理示例如何诱导分步解题能力。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Nate Nye et al. (2021), [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) — 用显式 scratchpad 保存中间计算，改进复杂推理。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Takeshi Kojima et al. (2022), [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) — “Let's think step by step” 作为零样本推理提示。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Long Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT 的监督微调、奖励模型和 PPO 三阶段 RLHF 流程。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Hyung Won Chung et al. (2022), [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — FLAN 展示跨任务指令微调的泛化能力。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Dan Hendrycks et al. (2021), [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) — MMLU 用多学科考试题评估广泛知识与推理。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Rishi Bommasani et al. (2022), [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — 讨论基础模型的能力、风险、偏差和评测维度。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Ronald J. Williams (1992), [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696) — REINFORCE policy-gradient 的经典推导。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Richard S. Sutton, Andrew G. Barto (2018), [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — RL 状态、价值、策略和策略梯度的标准教材。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Volodymyr Mnih et al. (2015), [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236) — DQN 将深度视觉表征与 Q-learning 结合。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- John Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 以 clipped surrogate objective 稳定策略更新，常用于 RLHF。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Paul F. Christiano et al. (2017), [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 用人类偏好训练奖励模型，再优化策略。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Nisan Stiennon et al. (2020), [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — 将 RLHF 应用于摘要并比较自动指标与人类偏好。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Yuntao Bai et al. (2022), [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — 用原则和 AI feedback 代替部分人工标注，训练更安全的助手。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）
- Geoffrey Irving, Amanda Askell (2019), [AI Safety via Debate](https://arxiv.org/abs/1805.00899) — 以对抗式辩论探索用 AI 监督复杂行为。（来源：`slides/cs224n-2023-lecture11-prompting-rlhf.pdf`）

## Slides 补充：问答、检索与多模态任务

- John Simmons et al. (1964), [The Semantic Structure of English and Its Application to Automatic Language Processing](https://doi.org/10.1109/TAI.1964.1308195) — 早期自动问答和语义解析系统的代表。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Matthew Richardson, Christopher J. C. Burges, Erin Renshaw (2013), [MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text](https://aclanthology.org/W13-1709/) — 以多选阅读理解评测机器文本理解。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Pranav Rajpurkar et al. (2016), [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 众包 span extraction 阅读理解基准。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Karl Moritz Hermann et al. (2015), [Teaching Machines to Read and Comprehend](https://arxiv.org/abs/1506.03340) — Attentive Reader 将注意力引入文档问答。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Danqi Chen, Christopher Manning (2016), [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/abs/1606.02858) — Stanford Attentive Reader 的神经阅读理解分析。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Shuohang Wang et al. (2017), [R-NET: Machine Reading Comprehension with Self-matching Networks](https://arxiv.org/abs/1706.04115) — 自匹配注意力和 gated recurrent reader。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Minjoon Seo et al. (2017), [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) — BiDAF 同时建模 query-to-context 和 context-to-query 注意力。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Caiming Xiong et al. (2017), [Dynamic Coattention Networks For Question Answering](https://arxiv.org/abs/1611.01604) — 动态 co-attention 和迭代答案指针。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Fengbin Chen et al. (2017), [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051) — DrQA 将文档检索和阅读器组合为开放域 QA。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Mandar Joshi et al. (2020), [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529) — 面向 span 预测优化预训练，提升 QA 和 coreference。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Robin Jia, Percy Liang (2017), [Adversarial Examples for Evaluating Reading Comprehension Systems](https://arxiv.org/abs/1707.07328) — 用自动构造扰动测试阅读理解系统是否真正理解。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Marco Tulio Ribeiro et al. (2020), [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118) — 用行为测试矩阵发现模型脆弱性。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Vladimir Karpukhin et al. (2020), [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — 双编码器 dense retriever 取代稀疏检索作为 RAG 基础。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Gautier Izacard, Edouard Grave (2021), [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282) — FiD 将多段检索证据分别编码后融合生成答案。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）
- Adam Roberts et al. (2020), [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/2002.08909) — 研究语言模型参数记忆与检索知识的关系。（来源：`slides/cs224n-2023-lecture12-QA.pdf`）

## Slides 补充：句子 CNN、树结构与语言学

- Ye Zhang, Byron Wallace (2016), [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820) — 系统分析句子 CNN 超参数和可复现实验细节。（来源：`slides/cs224n-2023-lecture13-CNN-TreeRNN.pdf`）
- Quoc V. Le, Tomas Mikolov (2014), [Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053) — Paragraph Vector 将文档级 ID 与词向量共同训练。（来源：`slides/cs224n-2023-lecture13-CNN-TreeRNN.pdf`）
- Karl Moritz Hermann, Phil Blunsom (2013), [The Role of Syntax in Vector Space Models of Compositional Semantics](https://arxiv.org/abs/1305.0741) — 用组合自编码器探讨句法结构对语义表示的影响。（来源：`slides/cs224n-2023-lecture13-CNN-TreeRNN.pdf`）
- Minwei Dong et al. (2014), [Learning to Parse and Generate with Recursive Neural Networks](https://aclanthology.org/D14-1162/) — 将递归神经网络用于句法和情感组合的早期工作。（来源：`slides/cs224n-2023-lecture13-CNN-TreeRNN.pdf`）
- Sven Schuster, Christopher D. Manning (2016), [Enhanced English Universal Dependencies: An Improved Representation for Natural Language Understanding Tasks](https://arxiv.org/abs/1609.02694) — 为语义任务扩展依存图表示。（来源：`slides/cs224n-2023-lecture14-insights-linguistics.pdf`）
- Noam Chomsky (1957), [Syntactic Structures](https://doi.org/10.1515/9783112316007) — 生成语法和层级句法结构的经典起点。（来源：`slides/cs224n-2023-lecture14-insights-linguistics.pdf`）
- Charles F. Hockett (1960), [The Origin of Speech](https://www.jstor.org/stable/1716276) — 以设计特征讨论人类语言与动物交流的差异。（来源：`slides/cs224n-2023-lecture14-insights-linguistics.pdf`）
- Edward Gibson et al. (2013), [How Efficiency Shapes Human Language](https://doi.org/10.1126/science.1238186) — 说明信息论效率如何塑造词序、形态和表达形式。（来源：`slides/cs224n-2023-lecture14-insights-linguistics.pdf`）
- Joseph Greenberg (1963), [Some Universals of Grammar with Particular Reference to the Order of Meaningful Elements](https://doi.org/10.1515/9783111350210-003) — 跨语言词序类型学的经典统计总结。（来源：`slides/cs224n-2023-lecture14-insights-linguistics.pdf`）
- Leonard Talmy (1985), [Force Dynamics in Language and Cognition](https://doi.org/10.1080/01690968508402115) — 用力动态解释语言中的因果、阻碍和事件结构。（来源：`slides/cs224n-2023-lecture14-insights-linguistics.pdf`）
- Ethan Chi, John Hewitt, Christopher D. Manning (2020), [Finding Universal Grammatical Relations in Multilingual BERT](https://arxiv.org/abs/2005.04511) — 比较多语言 BERT 是否学习跨语言共享的语法关系。（来源：`slides/cs224n-2023-lecture14-insights-linguistics.pdf`）

## Slides 补充：代码生成与程序推理

- Sumit Gulwani (2011), [Automating String Processing in Spreadsheets Using Input-Output Examples](https://doi.org/10.1145/1926385.1926393) — FlashFill 以示例合成程序，奠定示例驱动代码生成。（来源：`slides/cs224n-2023-lecture15-code-generation.pdf`）
- Noah D. Goodman, Michael C. Frank (2016), [Pragmatic Language Interpretation as Probabilistic Inference](https://doi.org/10.1016/j.tics.2016.08.005) — 以概率程序和语用推理解释人类如何从有限示例归纳规则。（来源：`slides/cs224n-2023-lecture15-code-generation.pdf`）
- Jacob Austin et al. (2021), [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) — 评估大语言模型生成、修复和解释程序的能力。（来源：`slides/cs224n-2023-lecture15-code-generation.pdf`）
- Yujia Li et al. (2022), [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814) — 用大规模采样、过滤和聚类解决竞赛编程题。（来源：`slides/cs224n-2023-lecture15-code-generation.pdf`）
- Aitor Lewkowycz et al. (2022), [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858) — Minerva 展示数学、科学题上的大模型推理和数据配方。（来源：`slides/cs224n-2023-lecture15-code-generation.pdf`）
- Shunyu Yao et al. (2023), [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 交替生成思考轨迹和工具动作，连接语言推理与执行。（来源：`slides/cs224n-2023-lecture15-code-generation.pdf`）
- Timo Schick et al. (2023), [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — 让语言模型学习何时调用外部 API。（来源：`slides/cs224n-2023-lecture15-code-generation.pdf`）

## Slides 补充：指代消解、探针与模型分析

- Kevin Clark, Christopher D. Manning (2016), [Deep Reinforcement Learning for Mention-Ranking in Coreference Models](https://arxiv.org/abs/1609.08667) — 用神经 mention-ranking 和强化学习改进指代消解。（来源：`slides/cs224n-2023-lecture17-coref.pdf`）
- Kenton Lee et al. (2017), [End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 直接在候选 span 上联合 mention detection 和 antecedent scoring。（来源：`slides/cs224n-2023-lecture17-coref.pdf`）
- Greg Durrett, Dan Klein (2013), [Easy-First Entity Resolution with Rich Linguistic Features](https://aclanthology.org/D13-1059/) — 以局部高置信决策构造实体和指代链。（来源：`slides/cs224n-2023-lecture17-coref.pdf`）
- Dipanjan Das, Noah A. Smith (2011), [Graph-Based Lexicon Expansion with a Sense-Constrained Random Walk](https://aclanthology.org/P11-1102/) — slides 用作图式语义消歧和结构推断的背景。（来源：`slides/cs224n-2023-lecture17-coref.pdf`）
- Tolga Bolukbasi et al. (2016), [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520) — 揭示词向量中的性别偏差并提出几何去偏。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）
- Christopher D. Manning et al. (2020), [Emergent linguistic structure in artificial neural networks trained by self-supervision](https://doi.org/10.1073/pnas.1907367117) — 讨论 Transformer/自监督网络内部如何形成词类、句法和指代结构。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）
- Tal Linzen et al. (2016), [Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies](https://arxiv.org/abs/1611.01368) — 用 subject–verb agreement 测试 RNN 是否学习语法依赖。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）
- Kevin Meng et al. (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) — ROME 通过定位 MLP 参数编辑模型事实记忆。（来源：`slides/Been-Kim-StanfordLectureMarch2023.pdf`, `slides/cs224n-2023-lecture18-analysis.pdf`）
- Eric Mitchell et al. (2022), [MEND: Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309) — 用元学习的梯度变换器高效修改模型知识。（来源：`slides/Been-Kim-StanfordLectureMarch2023.pdf`）
- John Hewitt, Percy Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) — 用控制任务判断 probing 得分是否只是探针容量造成。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）
- John Hewitt, Christopher D. Manning (2019), [A Structural Probe for Finding Syntax in Word Representations](https://arxiv.org/abs/1906.04238) — 以低秩变换从 contextual embeddings 恢复依存树几何。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）
- Jesse Vig et al. (2020), [Investigating Gender and Bias in GPT-3 with a Neural Network Interpretation Framework](https://arxiv.org/abs/2002.02015) — 用 attention 归因分析生成模型中的性别偏差。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）
- Paul Michel, Omer Levy, Graham Neubig (2019), [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) — 研究 Transformer 多头注意力的可剪枝性和功能冗余。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）
- Eric Wallace et al. (2019), [Universal Adversarial Triggers for Attacking and Analyzing NLP](https://arxiv.org/abs/1908.07125) — 学习可迁移的触发 token，揭示模型脆弱性。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）
- Belinkov, Bisk (2018), [Synthetic and Natural Noise Both Break Neural Machine Translation](https://arxiv.org/abs/1711.02173) — 证明字符级噪声会显著破坏 NMT，强调鲁棒性评测。（来源：`slides/cs224n-2023-lecture18-analysis.pdf`）

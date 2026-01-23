# Machine Learining Fundamentals

## Basics of ML - what is a neuron, neural network
*   **A. Jung (2022)** [Machine Learning: The Basics](https://arxiv.org/abs/1805.05052)

## Tokenization
*   **Sennrich, R., Haddow, B., & Birch, A. (2016).** [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/). *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*. (Foundational paper for Byte-Pair Encoding (BPE) and subword tokenization used in modern Transformers).
*   **Kudo, T., & Richardson, J. (2018).** [SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing](https://arxiv.org/abs/1808.06226). *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. (Used by DeBERTa and NLLB models).

## Regression vs Classification?
*   **Paltoglou, G., & Thelwall, M. (2010).** [A study of information retrieval weighting schemes for sentiment analysis](https://aclanthology.org/P10-1134/). *Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics*. (Discusses weighting schemes relevant for continuous scoring).

## Transformers
*   **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** [Attention is all you need](https://arxiv.org/abs/1706.03762). *Advances in neural information processing systems*, 30. (The original Transformer paper).
*   **He, P., Liu, X., Gao, J., & Chen, W. (2020).** [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654). *arXiv preprint arXiv:2006.03654*. (The architecture for fine-tuned model).
*   **He, P., Gao, J., & Chen, W. (2021).** [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543). *arXiv preprint arXiv:2111.09543*. (The specific V3 architecture used in zero-shot model).
*   **Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).** [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). *arXiv preprint arXiv:1910.01108*. (Relevant to distilbert models).

## Software & Frameworks
*   **Wolf, T., et al. (2020).** [Transformers: State-of-the-Art Natural Language Processing](https://aclanthology.org/2020.emnlp-demos.6/). *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. (The library powering the implementation).
*   **Paszke, A., et al. (2019).** [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703). *Advances in Neural Information Processing Systems*, 32. (The backend framework used).

## Text analysis
*   **Taboada, M., Brooke, J., Tofiloski, M., Voll, K., & Stede, M. (2011).** [Lexicon-based methods for sentiment analysis](https://aclanthology.org/J11-2011/). *Computational Linguistics*, 37(2), 267-307. (Core theory for lexicon approaches).
*   **Hutto, C. J., & Gilbert, E. (2014).** [VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text](https://ojs.aaai.org/index.php/ICWSM/article/view/14550). *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1). (Source of VADER implementation).
*   **Brown, T., et al. (2020).** [Language models are few-shot learners](https://arxiv.org/abs/2005.14165). *Advances in neural information processing systems*, 33. (GPT-3 paper).
*   **Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, others (2025)** [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](https://arxiv.org/abs/2507.06261)

## Fine-tuning models
*   **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). *arXiv preprint arXiv:1810.04805*. (Standard reference for the pre-training/fine-tuning paradigm).
*   **Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019).** [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583) *Chinese Computational Linguistics*.

# Language

## What is objectivity
*   **Pang, B., & Lee, L. (2008).** [Opinion mining and sentiment analysis](https://doi.org/10.1561/1500000011). *Foundations and Trends® in Information Retrieval*, 2(1–2), 1-135. (The seminal survey defining the field).
*   **Wiebe, J., & Riloff, E. (2005).** [Creating subjective and objective sentence classifiers from unannotated text](https://link.springer.com/chapter/10.1007/978-3-540-30586-6_53). *Computational Linguistics and Intelligent Text Processing*.

## How to measure objectivity
*   **Wilson, T., Wiebe, J., & Hoffmann, P. (2005).** [Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis](https://aclanthology.org/H05-1044/). *Proceedings of HLT/EMNLP 2005*. (Source of the MPQA Subjectivity Lexicon).
*   **Gilardi, F., Alizadeh, M., & Kubli, M. (2023).** [ChatGPT outperforms crowd workers for text-annotation tasks](https://doi.org/10.1073/pnas.2305016120). *Proceedings of the National Academy of Sciences*, 120(30). (Justifies using LLMs for labeling/evaluation).

## LLM-assisted translation
*   **NLLB Team (2022)**[No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)

# Data Processing & Bias

## Dataset Cleaning
*   **Barik, A., & Das, D. (2024).** [A Comprehensive Survey of Text Data Cleaning Techniques: Challenges, Methods, and Best Practices](https://jsaer.com/download/0107-1-155-177.pdf). *Journal of Scientific and Engineering Research*, 11(7), 155-177. (Overview of standard cleaning practices for NLP).
*   **Wang, Y., Liu, Y., & Liu, Y. (2023).** [Multi-News+: Cost-efficient Dataset Cleansing via LLM-based Data Annotation](https://arxiv.org/abs/2305.14251). *arXiv preprint arXiv:2305.14251*. (Relevant for using LLMs to clean news datasets).
*   **Bhattacharjee, M., et al. (2025).** [NewsSumm: The World's Largest Human-Annotated Multi-Document News Summarization Dataset for Indian English](https://www.mdpi.com/2306-5729/10/2/39). *Data*, 10(2), 39. (Discusses specific cleaning pipelines for news articles: deduplication, date standardization).

## Media Bias Detection

*   **Hamborg, F. (2020).** [Media Bias, the Social Sciences, and NLP: Automating Frame Analyses to Identify Bias by Word Choice and Labeling](https://aclanthology.org/2020.acl-main.718/). *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. (Crucial paper for understanding "bias by word choice," which aligns with subjectivity analysis).

*   **Gummalam, N., & Greenberg, C. (2025).** [Bias Detection in Media: An NLP-Based Approach using Corpus Statistics and Sentence Embeddings](https://researchgate.net/publication/387920409_Bias_Detection_in_Media_An_NLP-Based_Approach_using_Corpus_Statistics_and_Sentence_Embeddings). *Journal of Student Research*, 14(1). (Recent approach using embeddings and statistics).

*   **Spinde, T., et al. (2021).** [Automated identification of media bias in news articles: an interdisciplinary literature review](https://link.springer.com/article/10.1007/s00799-021-00302-y). *International Journal on Digital Libraries*, 22, 235–250.



## LLMs in Data Creation & Labeling



*   **Tan, Z., et al. (2024).** [Large Language Models for Data Annotation: A Survey](https://arxiv.org/abs/2402.13446). *arXiv preprint arXiv:2402.13446*. (Comprehensive overview of using LLMs as annotators).



*   **Wang, Y., et al. (2023).** [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560). *arXiv preprint arXiv:2212.10560*. (Seminal paper on generating synthetic training data from LLMs).



*   **He, X., et al. (2023).** [AnnLLM: Using Large Language Models for Active Learning-based Data Annotation](https://arxiv.org/abs/2311.10080). *arXiv preprint arXiv:2311.10080*. (Methodology for efficient labeling using LLMs).



*   **Honovich, O., et al. (2022).** [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689). *arXiv preprint arXiv:2212.09689*. (Demonstrates the power of synthetic datasets).

## Web Scraping & Anti-Bot Evasion
*   **Kincaid, J. P., et al. (1975).** [Derivation of New Readability Formulas (Automated Readability Index, Fog Count and Flesch Reading Ease Formula) for Navy Enlisted Personnel](https://stars.library.ucf.edu/istlibrary/56/). *Research Branch Report 8-75, Chief of Naval Technical Training*. (The foundational theory behind readability extraction algorithms).
*   **Uzun, E., et al. (2014).** [The anatomy of a large-scale web crawler](https://ieeexplore.ieee.org/document/6906803). *IEEE International Congress on Big Data*. (Discusses crawler architecture and challenges).
*   **Zhao, B., et al. (2017).** [Anti-crawling techniques: A survey](https://ieeexplore.ieee.org/document/7973685). *IEEE International Conference on Big Data and Smart Computing*. (Academic overview of the protections tools like Cloudscraper aim to bypass).




*   **P. Boruch (2021)** [Problem obiektywizmu przekazów medialnych a paski informacyjne publikowane w głównych serwisach informacyjnych Polsatu i TVP](https://academic-journals.eu/pl/download?path=%2Fuploads%2FZm9sZGVycHVibWVkaWE0OQ%3D%3D%2Fdocuments%2Fhumcultstudies-2-4-6.pdf). *Humanities and Cultural Studies*, 2(4).
*   **Demagog (2024)** [Obiektywizm w mediach to już przeszłość](https://www.facebook.com/DEMAGOG/posts/-obiektywizm-w-mediach-to-już-przeszłośćjak-wskazuje-dr-hab-adam-szynol-mediozna/692430713416141/). *Facebook Post*.
*   **Press.pl (2023)** [Propagandysta TVP nauczał o dziennikarskim obiektywizmie. Odrobiliśmy tę lekcję i pytamy - czym jest obiektywizm](https://www.press.pl/tresc/76915,propagandysta-tvp-nauczal-o-dziennikarskim-obiektywizmie_-odrobilismy-te-lekcje-i-pytamy_-czym-jest-obiektywizm). *Press.pl*.
*   **P. Płuska (2024)** [Obiektywizm to nie statystyki, ale prawo do kontrargumentu](https://neptuntv.ug.edu.pl/2024/12/14/obiektywizm-to-nie-statystyki-ale-prawo-do-kontrargumentu-wywiad-z-pawlem-pluska-szefem-1930/). *Neptun TV*.
*   **M. Gach (2020)** [Dziennikarstwo obiektywne, subiektywne czy partyjne?](https://histmag.org/Dziennikarstwo-obiektywne-subiektywne-czy-partyjne-20678). *Histmag.org*.
*   **J. Pinkas (2024)** [Medyczne fake newsy rozprzestrzeniają się jak choroby zakaźne: szybko i łatwo](https://www.onkonet.pl/n_n_medyczne_fake_news.php). *Onkonet.pl*.
*   **M. Rech (2023)** [Epidemia fake newsów sama się nie skończy. A jest tak samo groźna jak zmiany klimatu](https://wyborcza.pl/7,179012,29355646,epidemia-fake-newsow-sam-sie-nie-skonczy-a-jest-tak-samo-grozna.html). *Wyborcza.pl*.
*   **Rynek Zdrowia (2024)** [Wirus HMPV pożywką dla fake newsów. Eksperci wyjaśniają, czy grozi nam lockdown](https://www.rynekzdrowia.pl/Choroby-zakazne/Wirus-HMPV-pozywka-dla-fake-newsow-Eksperci-wyjasniaja-czy-grozi-nam-lockdown,267523,22.html). *Rynek Zdrowia*.
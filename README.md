# Multi-dimensional Texture Tactile Reproduction Method based on Generative Replay Incremental Learning Model

<h3 align="center">ABSTRACT</h3>

With the continuous expansion of texture sample data, texture tactile models need to maintain stable rendering of learned textures while continuously acquiring new texture features. However, existing models often fail to effectively absorb new knowledge in continuous learning scenarios, or cannot simultaneously balance the learning efficiency of new textures with the maintenance of existing reproduction capabilities. To address this issue, we propose a Texture Tactile Continuous Learning Model (TTCLM) based on generative replay mechanism. This model has added a continuous learning phase on the basis of our initial training phase work. During the continuous learning phase, buffers and adapters are introduced for replay learning, and the backbone network parameters are frozen to avoid new tasks damaging existing knowledge. Through the joint loss optimization of the decoder and discriminator, TTCLM has improved its fitting ability for new tasks. We trained the model on the SENS3 tactile dataset and conducted three systematic validation experiments. The results showed that TTCLM exhibited excellent performance in continuous learning and resistance to catastrophic forgetting (BWT value of -0.101), and demonstrated good robustness under changes in the order of new tasks. Finally, we conducted three user experiments, and the results showed that the proposed tactile continuous learning method can effectively absorb new sample information, enabling users to achieve higher accuracy (88.33%) in virtual and real texture matching tasks. Compared with existing baseline methods, this method achieved the highest perceived average similarity score (7.60), verifying its continuous learning ability and user experience improvement effect in dynamic texture scenes.

<h3 align="center">Texture Tactile Continuous Learning Model</h3>

In the preliminary work [Chen D, Ding Y, Gao P, et al. Multi dimensional Texture Haptic Cross modal Generation and Display Method based on Bi Mamba Network [J]. IEEE Transactions on Instrumentation and Measurement, 2025], we constructed a multidimensional texture tactile rendering model based on multimodal data (texture images and user interaction information) to generate real-time acceleration and friction signals for user interaction with virtual textures. On this basis, this study made structural improvements and introduced a continuous learning strategy based on replay mechanism. The overall architecture is shown in the following figure.

<p align="center">
  <img src="fig1.png" alt="fig1" width="900"/>
</p>

### Dataset
The SENS3 dataset [Balasubramanian J K, Kodak B L, Vardar Y. Sens3: Multisensory database of finger surface interactions and corresponding sensing [C]//International Conference on Human Haptic Sensing and Touch enabled Computer Applications. Cham: Springer Nature Switzerland, 2024: 262-277.] is a multimodal texture tactile perception resource library designed specifically for object surface interaction scenarios. Its core design revolves around the physical logic of real tactile interaction. 
This study focuses on the multidimensional tactile modeling task of textures in dynamic interactive environments, focusing on two core dimensions: the continuous learning ability of new category textures and the anti forgetting ability of old category textures. We select fabric, wood, metal, plastic, and rubber in the SENS3 database as the basic texture category training model, and then take sandpaper, paper, leather, foam, and synthetic materials as the incremental texture categories to build the domain incremental learning scene in turn. For more details, see https://link.springer.com/chapter/10.1007/978-3-031-70058-3_21

### Data Preparation
Start by running `utils/data_deal.py` to retrieve and organize the required original dataset.
```bash
utils/data_deal.py
```

Then run 'utilities/dataprocess. py', align the Excel data by file name, and merge it into a CSV file. Crop the data before and after and save it as standardized results to obtain the initial training data and incremental learning data, which are two folders.
```bash
utils/dataprocess.py
```






(a) Initial training phase
In the initial training phase, the system receives multimodal input data from the initial samples, including texture images, pressing force, and sliding speed. Texture images are extracted with high-level visual features using ResNet50 network, and their dimensions are adjusted to be consistent with temporal modes through linear layer mapping. The interactive action signals (based on pressure and sliding speed) are segmented and normalized through sliding windows to ensure comparability and stability of the temporal inputs. Subsequently, multi-modal features are fused in parallel and input into the Bi Mamba encoder for deep feature extraction. Finally, the deep features are mapped back to the temporal space through a decoder to generate acceleration and friction signals, completing the training of the initial tactile rendering network and constructing the initial pre training model.






(b) Incremental learning stage
To cope with the continuous addition of new texture samples and dynamic changes in interactive scenes in practical applications, we further constructed an incremental learning stage. At this stage, the model is synchronously trained through the joint input of replay samples in the buffer and newly added texture samples. To optimize the incremental learning process, we introduced adapter and discriminator modules. The adapter enhances the model's adaptability to new data distributions, allowing it to maintain stable perception of learned textures while learning new texture features. By updating the lightweight parameters of the adapter, it is possible to effectively avoid damaging the learned knowledge representation in the backbone network, thereby significantly alleviating catastrophic forgetting. At the same time, the discriminator distinguishes and constrains the authenticity of the output during the generation of tactile signals, further improving the accuracy and physical consistency of tactile signals. In order to maintain the stability of learned features, the feature extraction module and encoder parameters are frozen at this stage, allowing only newly added lightweight structures to participate in training, thereby avoiding disturbing the learned texture features and tactile mapping relationship. Through the collaborative effect of cache, adapter, and discriminator, the model can still access old task data while learning new tasks, effectively reducing the risk of forgetting caused by distribution drift, and maintaining stable memory of old knowledge while continuously learning new knowledge, significantly enhancing the stability and anti forgetting ability of the model in the long-term learning process.
On this basis, an adapter module is introduced to learn finite characteristic parameters in dynamic tactile data, and to achieve adaptive updates of the model to the distribution of new input data and dynamic interaction scenarios without damaging the pre training weights of the backbone network. Specifically, the adapter is embedded between the pre trained base layer (feature extraction module and encoder) and the task specific layer (decoder and discriminator), capturing the difference information between different data distributions through a small number of trainable parameters, thereby enhancing its plasticity and generalization ability to dynamic inputs while maintaining model stability.




Introduction
=================

### 前言＆廢話
7月初我從原本的公司離職, 耍廢1,2週調整心情之後, 決定開始學習之前再工作時一直挺感興趣的deep learning領域.
目前學習使用python & tensorflow大約2個禮拜了, 原本是偷懶沒有打算寫筆記, 但因為deep learning實在有太多需要注意的點跟有趣的東西, 因此才開始使用這個筆記來紀錄一些使用心得以及目前開發的各種紀錄

目前給自己的目標大致上是：

- 短期計畫(7月中~7月底):
  - 學習使用python & tensorflow (Keras)
- 中期計畫(8月~9月底):
  - 完成一個多車道檢測的模型
  - 將完成的模型透過SNPE SDK轉換至Qualcomm 835上運行
  - Lane marking labeling Tool 


### 學習使用python & tensorflow (Keras)
7月中開始我開始學習python & keras的開發, 之前工作與唸書時都是使用c/c++, 一開始使用python到是碰到不少問題, 目前python的學習到是比較像是邊撞牆邊修正, 幸好IDE跟google都提供了許多幫忙, 目前大致能寫出一個簡單的python程式.

目前python使用到的部份基本上還是跟tensorflow相關, 所以涉略的部份非常的少... 目前是預計除了基本寫法之外, 加入一個lane marking labeling tool來作為gui相關操作的練習, 但目前對於多車道檢測還處於探索期, 打算先使用現有的dataset, 例如TuSimple, CULane...等, 先了解一下領域內資料集的規範, 接著再決定使用的格式等等

### Multi Lane Detection
以下是我目前開發多車道檢測的各種心得與想法
- <b>Multi Lane Detector</b> 
  - [Segmentation based](./MultiLane_notes/segmentation_based.md) 
  - [Polynomial Curve Fitting based](./MultiLane_notes/segmentation_based.md)
  - [Perspective & Segmentation based](./MultiLane_notes/segmentation_based.md)


### Paper notes
以下是我對各種論文的閱讀筆記或想法
- <b>[Mobilenet v1](./paper_notes/mobilenet_v1.md)</b> 
  - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
  - [paper](https://arxiv.org/abs/1704.04861)

- <b>[Mobilenet v2](./paper_notes/mobilenet_v2.md)</b> 
  - MobileNetV2: Inverted Residuals and Linear Bottlenecks
  - [paper](https://arxiv.org/pdf/1801.04381.pdf)

- <b>[Mobilenet v3](./paper_notes/mobilenet_v3.md)</b> 
  - Searching for MobileNetV3
  - [paper](https://arxiv.org/abs/1905.02244)

- <b>[Densenet](./paper_notes/Densenet.md)</b> 
  - Densely Connected Convolutional Networks
  - [paper](https://arxiv.org/abs/1608.06993)

- <b>[Yolo v1](./paper_notes/yolo_v1.md)</b> 
  - You Only Look Once: Unified, Real-Time Object Detection
  - [paper](https://arxiv.org/abs/1506.02640)

- <b>[SENet](./paper_notes/Squeeze_and_Excitation.md)</b> 
  - Squeeze-and-Excitation Networks
  - [paper](https://arxiv.org/abs/1709.01507)










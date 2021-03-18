# 简单实验
## 优化器选择
### sgd
sgd 优化器的学习率设置为0.01 dropout 0.2  
![SGD auc](sgd_auc.png)  
![SGD loss](sgd_loss.png)  

### adam
adam 优化器的学习率设置为0.0001,dropout 0.2  
![Adam auc](adam_auc.png)  
![Adam loss](adam_loss.png)  
### 总结
由于推荐算法需要很快的训练时间,所以日常使用模型是,为了快速迭代,故选择adam作为常用优化器,在10+个epoch就能取得不错的性能.

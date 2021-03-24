# Bert-BiLSTM-CRF-pytorch
使用谷歌预训练bert做字嵌入的BiLSTM-CRF序列标注模型

本模型使用谷歌预训练bert模型（https://github.com/google-research/bert）， 
同时使用pytorch-pretrained-BERT（https://github.com/huggingface/pytorch-pretrained-BERT）项目加载bert模型并转化为pytorch参数，

```
export TF_BERT_BASE_DIR=~/Models/roberta_wwm_large
export PT_BERT_BASE_DIR=~/Models/roberta_pytorch

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
	$TF_BERT_BASE_DIR/bert_model.ckpt \
	$TF_BERT_BASE_DIR/bert_config.json \
	$PT_BERT_BASE_DIR/pytorch_model.bin
	
cp $TF_BERT_BASE_DIR/bert_config.json $PT_BERT_BASE_DIR/bert_config.json
cp $TF_BERT_BASE_DIR/vocab.txt $PT_BERT_BASE_DIR/vocab.txt
```

CRF代码参考了SLTK（https://github.com/liu-nlper/SLTK）

准备数据格式参见data

模型参数可以在config中进行设置

运行代码

python main.py train --use_cuda=False --batch_size=10  



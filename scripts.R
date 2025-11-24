# 代码简介：10种具有变量筛选和预后模型构建的算法组合
# 
#算法输入：
#训练集表达谱：行为特征(如SYMBOL/ENSEMBL等，但需与测试集有一定交集)，列为样本的表达矩阵，格式见InputData文件中Train_expr.txt
#训练集生存信息：行为样本，列包括生存状态以及生存时间(代码中需分别指定相对应的变量名)，格式见InputData文件中Train_surv.txt
#测试集表达谱：行为特征(如SYMBOL/ENSEMBL等，但需与训练集有一定交集)，列为样本的表达矩阵，格式见InputData文件中Test_expr.txt
#测试集生存信息：行为样本，列包括生存状态以及生存时间，以及一列用于指定队列信息的变量，格式见InputData文件中Test_surv.txt
#注意：建议对表达谱进行预筛选，保留几十至几千量级的感兴趣特征用于最优模型筛选，否则运行时间将大大延长

#本代码适用于：
#存在较多预后意义不明的变量，需同时进行变量筛选，模型筛选和预后模型构建的场景

#代码输出结果包括：
#101种算法在训练集上获得的模型(model.rds)，在测试集(或包括训练集)上的评估结果(Cindex_mat)，以及评估结果的热图展示(Cindex.pdf)。

# 设置工作路径
work.path <- "C:\\101jqxx"; setwd(work.path) 

# 设置其他路径
code.path <- file.path(work.path, "Codes")
data.path <- file.path(work.path, "InputData")
res.path <- file.path(work.path, "Results")
fig.path <- file.path(work.path, "Figures")

# 如不存在这些路径则创建路径
if (!dir.exists(data.path)) dir.create(data.path)
if (!dir.exists(res.path)) dir.create(res.path)
if (!dir.exists(fig.path)) dir.create(fig.path)
if (!dir.exists(code.path)) dir.create(code.path)

# BiocManager::install("mixOmics")
# BiocManager::install("survcomp")
# devtools::install_github("binderh/CoxBoost")
# install.packages("randomForestSRC")
# install.packages("snowfall")

# 加载需要使用的R包
library(openxlsx)
library(seqinr)
library(plyr)
library(survival)
library(randomForestSRC)
library(glmnet)
library(plsRcox)
library(superpc)
library(gbm)
library(mixOmics)
library(survcomp)
library(CoxBoost)
library(survivalsvm)
library(BART)
library(snowfall)
library(ComplexHeatmap)
library(RColorBrewer)

# 加载模型训练以及模型评估的脚本
source(file.path(code.path, "ML.R"))

## Training Cohort ---------------------------------------------------------
# 训练集表达谱是行为基因（感兴趣的基因集），列为样本的表达矩阵（基因名与测试集保持相同类型，如同为SYMBOL或ENSEMBL等）
Train_expr <- read.table(file.path(data.path, "Training_expr.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
Train_expr <- Train_expr[rowSums(Train_expr > 0) > ncol(Train_expr) * 0.1, ] # 剔除大量无表达的基因，以免建模过程报错
# 训练集生存数据是行为样本，列为结局信息的数据框
Train_surv <- read.table(file.path(data.path, "Training_surv.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
Train_surv <- Train_surv[Train_surv$OS.time > 0, c("OS", "OS.time")] # 提取OS大于0的样本
comsam <- intersect(rownames(Train_surv), colnames(Train_expr))
Train_expr <- Train_expr[,comsam]; Train_surv <- Train_surv[comsam,,drop = F]

## Validation Cohort -------------------------------------------------------
# 测试集表达谱是行为基因（感兴趣的基因集），列为样本的表达矩阵（基因名与训练集保持相同类型，如同为SYMBOL或ENSEMBL等）
Test_expr <- read.table(file.path(data.path, "Testing_expr.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
# 测试集生存数据是行为样本，列为结局信息的数据框
Test_surv <- read.table(file.path(data.path, "Testing_surv.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
Test_surv <- Test_surv[Test_surv$OS.time > 0, c("Coho","OS", "OS.time")] # 提取OS大于0的样本
comsam <- intersect(rownames(Test_surv), colnames(Test_expr))
Test_expr <- Test_expr[,comsam]; Test_surv <- Test_surv[comsam,,drop = F]

# 提取相同基因
comgene <- intersect(rownames(Train_expr),rownames(Test_expr))
Train_expr <- t(Train_expr[comgene,]) # 输入模型的表达谱行为样本，列为基因
Test_expr <- t(Test_expr[comgene,]) # 输入模型的表达谱行为样本，列为基因

# Model training and validation -------------------------------------------

## method list --------------------------------------------------------
# 此处记录需要运行的模型，格式为：算法1名称[算法参数]+算法2名称[算法参数]
# 目前仅有StepCox和RunEnet支持输入算法参数
methods <- read.xlsx(file.path(code.path, "41467_2022_28421_MOESM4_ESM.xlsx"), startRow = 2)
methods <- methods$Model
methods <- gsub("-| ", "", methods)
head(methods)

## Train the model --------------------------------------------------------
model <- list()
set.seed(seed = 123)
for (method in methods){
  cat(match(method, methods), ":", method, "\n")
  method_name = method # 本轮算法名称
  method <- strsplit(method, "\\+")[[1]] # 各步骤算法名称
  
  Variable = colnames(Train_expr) # 最后用于构建模型的变量
  for (i in 1:length(method)){
    if (i < length(method)){
      selected.var <- RunML(method = method[i], # 机器学习方法
                            Train_expr = Train_expr, # 训练集有潜在预测价值的变量
                            Train_surv = Train_surv, # 训练集生存数据
                            mode = "Variable",       # 运行模式，Variable(筛选变量)和Model(获取模型)
                            timeVar = "OS.time", statusVar = "OS") # 用于训练的生存变量，必须出现在Train_surv中
      if (length(selected.var) > 5) Variable <- intersect(Variable, selected.var)
    } else {
      model[[method_name]] <- RunML(method = method[i],
                                    Train_expr = Train_expr[, Variable],
                                    Train_surv = Train_surv,
                                    mode = "Model",
                                    timeVar = "OS.time", statusVar = "OS")
    }
  }
}
saveRDS(model, file.path(res.path, "model.rds"))

## Evaluate the model -----------------------------------------------------

# 读取已报错的模型列表
model <- readRDS(file.path(res.path, "model.rds"))
summary(Train_expr)
summary(Test_expr)


Train_expr <- scale(Train_expr)
Test_expr <- scale(Test_expr)

# 对各模型计算C-index
#is.finite(Test_expr)
#Test_expr[!is.finite(Test_expr)] <- NA
Cindexlist <- list()
for (method in methods){
  Cindexlist[[method]] <- RunEval(fit = model[[method]], # 预后模型
                                  Test_expr = Test_expr, # 测试集预后变量，应当包含训练集中所有的变量，否则会报错
                                  Test_surv = Test_surv, # 训练集生存数据，应当包含训练集中所有的变量，否则会报错
                                  Train_expr = Train_expr, # 若需要同时评估训练集，则给出训练集表达谱，否则置NULL
                                  Train_surv = Train_surv, # 若需要同时评估训练集，则给出训练集生存数据，否则置NULL
                                  Train_name = "TCGA", # 若需要同时评估训练集，可给出训练集的标签，否则按“Training”处理
                                  cohortVar = "Coho", # 重要：用于指定队列的变量，该列必须存在且指定[默认为“Cohort”]，否则会报错
                                  timeVar = "OS.time", # 用于评估的生存时间，必须出现在Test_surv中
                                  statusVar = "OS") # 用于评估的生存状态，必须出现在Test_surv中
}
Cindex_mat <- do.call(rbind, Cindexlist)
write.table(Cindex_mat, file.path(res.path, "Cindex_mat.txt"),
            sep = "\t", row.names = T, col.names = T, quote = F)

# Plot --------------------------------------------------------------------

Cindex_mat <- read.table(file.path(res.path, "Cindex_mat.txt"),sep = "\t", row.names = 1, header = T,check.names = F,stringsAsFactors = F)
avg_Cindex <- apply(Cindex_mat, 1, mean)           # 计算每种算法在所有队列中平均C-index
avg_Cindex <- sort(avg_Cindex, decreasing = T)     # 对各算法C-index由高到低排序
Cindex_mat <- Cindex_mat[names(avg_Cindex), ]      # 对C-index矩阵排序

avg_Cindex <- as.numeric(format(avg_Cindex, digits = 3, nsmall = 3)) # 保留三位小数


if(ncol(Cindex_mat) < 3) { # 如果用于绘图的队列小于3个
  CohortCol <- c("red","blue") # 则给出两个颜色即可（可自行替换颜色）
} else { # 否则通过brewer.pal赋予超过3个队列的颜色
  CohortCol <- brewer.pal(n = ncol(Cindex_mat), name = "Paired") # 设置队列颜色
}
names(CohortCol) <- colnames(Cindex_mat)

cellwidth = 1; cellheight = 0.5
hm <- SimpleHeatmap(Cindex_mat, # 主矩阵
                    avg_Cindex, # 侧边柱状图
                    CohortCol, "steelblue", # 列标签颜色，右侧柱状图颜色
                    cellwidth = cellwidth, cellheight = cellheight, # 热图每个色块的尺寸
                    cluster_columns = F, cluster_rows = F) # 是否对行列进行聚类

pdf(file.path(fig.path, "Cindex2.pdf"), width = cellwidth * ncol(Cindex_mat) + 8, height = cellheight * nrow(Cindex_mat) * 0.45)
draw(hm)
invisible(dev.off())


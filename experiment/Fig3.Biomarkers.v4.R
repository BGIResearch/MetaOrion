library(RColorBrewer)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(ggbeeswarm)
library(tidyverse)
setwd('~/Documents/Meta Index/Result/features_20251219/')

abbreviate_taxa <- function(taxa_names) {
  abbreviations <- sapply(taxa_names, function(taxa) {
    words <- strsplit(taxa, " ")[[1]]
    if (length(words) >= 2) {
      # 取第一个单词的首字母，加上空格，加上第二个单词
      # 如果有更多单词，也加上（如用户例子中的"Prevotella copri clade a"）
      if (!grepl('Ggb',words[1])){
        genus_initial <- substr(words[1], 1, 1)
        if (length(words) == 2) {
          return(paste0(genus_initial, ". ", words[2]))
        } else {
          # 对于像"Prevotella copri clade a"这样的名字
          return(paste0(genus_initial, ". ", paste(words[-1], collapse = " ")))
        }
      } else {
        return(taxa)
      }
    } else {
      # 如果只有一个单词，返回原样
      return(taxa)
    }
  })
  return(abbreviations)
}

biomarker.stat <- function(features.info,rank,topN=30,col.grads=c("#FEE0D2","#CB181D")){ 
  idx <- rowSums(features.info[,grepl(rank,colnames(features.info))]<=topN,na.rm = T)>0
  rank.matrix <- features.info[idx,grepl(rank,colnames(features.info)),drop=F]
  rank.matrix[rank.matrix>topN] <- NA
  shared.count <- data.frame(shared_labels=rowSums(!is.na(rank.matrix)))
  shared.count$shared_taxa <- rownames(shared.count)
  summary_data <- shared.count %>%
    group_by(shared_labels) %>%
    summarise(
      count = n(),
      taxa_list = list(shared_taxa)  # 保存菌名列表
    ) %>% arrange(desc(shared_labels))
  
  summary_data <- summary_data %>%
    mutate(
      # 缩写菌名
      abbreviated_taxa = lapply(taxa_list, function(x) {
        if (length(x) > 0) {
          return(paste(abbreviate_taxa(x), collapse = ", "))
        } else {
          return("")
        }
      }),
      # 创建标注文本：菌种数量少于5时才显示菌名
      label_text = ifelse(count < 5 & shared_labels>5, 
                          paste0(as.character(count),'  ',as.character(abbreviated_taxa)),
                          as.character(count))
    )
  
  max_count <- max(summary_data$count)
  
  p <- ggplot(summary_data, aes(x = count, 
                                y = factor(shared_labels, levels = rev(sort(unique(shared_labels)))),
                                fill = log10(count))) +
    
    # 绘制深灰色柱子
    geom_col(width = 0.8) +
    
    # 添加文本标注
    # 对于菌种数量少于5的，显示菌名缩写
    # 对于菌种数量多于等于5的，显示数字
    geom_text(aes(label = label_text),
              hjust = 0,
              size = 4,
              color = "black",
              fontface = "italic") +
    # 设置 x 轴范围和刻度
    scale_x_continuous(
      position = "top",
      name = "Number of shared biomarkers",
      limits = c(0, max_count),           # 设置 x 轴范围
      expand = expansion(mult = c(0, 0.1))  # 底部不留扩展，顶部留10%空间
    )+
    
    scale_fill_gradient(
      low = col.grads[1],
      high = col.grads[2]
    ) +
    
    # 添加标签和标题
    labs(
      #title = "Distribution of shared biomarkers",
      x = "Shared taxa",
      y = "Number of Shared Conditions"
    ) +
    
    # 主题设置，模仿图片风格
    theme_classic() + theme(legend.position = "none")
  
  #p # export pdf 3.97x4.17
  return(list(shared.count,p))
}





features.heatmap <- function(features.info, annotations.colors,rank='.p.rank',topN=30,prev=2,show.effect.sign=T,rank.matrix=NULL){
  if(is.null(rank.matrix)){
    rank.matrix <- features.info[,grepl(rank,colnames(features.info)),drop=F]
    rank.matrix <- rank.matrix[rowSums(rank.matrix<(topN+1) & !is.na(rank.matrix)) >=prev,]
  }
  
  is.multi.col <- ncol(rank.matrix)>1
  weight.matrix <- features.info[rownames(rank.matrix),gsub(rank, ".attribution", colnames(rank.matrix)),drop=F]
  colnames(weight.matrix) <- sub('.attribution','',colnames(weight.matrix))
  
  if (!is.multi.col){
    weight.matrix <- weight.matrix[order(weight.matrix[,1],decreasing = F),,drop=F]
    rank.matrix <- rank.matrix[rownames(weight.matrix),,drop=F]
  }
  
  selected.annotation.colors <- annotation.colors$Phylum[unique(features.info[rownames(rank.matrix),'phylum'])]
  selected.annotation.colors <- list(Phylum=selected.annotation.colors)
  
  marker.p <- features.info[rownames(rank.matrix),grepl('p.value',colnames(features.info)),drop=F]
  marker.sign <- features.info[rownames(rank.matrix),grepl('effect.size',colnames(features.info)),drop=F]
  marker.sign[marker.p>0.05] <- NA
  sign.matrix <- marker.sign
  sign.matrix[marker.sign>0] <- '-'
  sign.matrix[marker.sign<0] <- '+'
  sign.matrix[is.na(sign.matrix)] <- ''
  sign.matrix[rank.matrix>topN & !is.na(rank.matrix)] <- '' # 去掉非topN以内的正负号
  rank.matrix.sign <- rank.matrix
  rank.matrix.sign[rank.matrix.sign>topN | is.na(rank.matrix.sign)] <- ''
  rank.matrix[rank.matrix>topN | is.na(rank.matrix)] <- -topN
  weight.matrix[is.na(weight.matrix)] <- 0 
  
  if (show.effect.sign){
    rank.matrix.sign <- as.data.frame(
      matrix(paste(as.matrix(rank.matrix.sign),as.matrix(sign.matrix),sep =''),nrow = nrow(sign.matrix)))
  }
  
  mat2_colors <- colorRampPalette(c("blue", "white", "red"))(100) 
  
  custom_breaks <- seq(-.5, .5, length.out = 100)
  
  #pheatmap::pheatmap(rank.matrix,cluster_rows = T,cluster_cols = is.multi.col,display_numbers = rank.matrix.sign,fontsize_number = 8, color = mat2_colors)
  p <- pheatmap::pheatmap(weight.matrix,cluster_rows = is.multi.col,cluster_cols = is.multi.col,display_numbers = rank.matrix.sign,fontsize_number = 8, color = mat2_colors,
                          annotation_row = phylum.info,
                          annotation_colors = selected.annotation.colors,breaks = custom_breaks)
  p
  if (is.multi.col){
    return(list(p,weight.matrix,rank.matrix.sign))
  }
}


# heatlhy
features.info <- read.csv('merged_all_labels_cleaned.csv',row.names = 1)
features.info <- features.info[,!grepl('Others',colnames(features.info))]
features.info <- features.info[,!grepl('pandisease',colnames(features.info))]

prev.cor <- data.frame(disease=NULL,feature.type=NULL,sample.type=NULL,rho=NULL,p=NULL)
abun.cor <- prev.cor
for (i in 1:15){
  idx <- colnames(features.info)[9*(i-1)+1]
  print(idx)
  x <- features.info[,grepl(sub('.attribution','',idx),colnames(features.info))]
  x <- x[!is.na(x[,1]),]
  print(dim(x))
  n.idx <- x[,1]<0
  n4 <- cor.test(x[n.idx,1],x[n.idx,4],method = 'spearman')
  n6 <- cor.test(x[n.idx,1],x[n.idx,6],method = 'spearman')
  p4 <- cor.test(x[!n.idx,1],x[!n.idx,4],method = 'spearman')
  p6 <- cor.test(x[!n.idx,1],x[!n.idx,6],method = 'spearman')
  
  if(!grepl('Melanoma',idx)){ # melanoma only case
    n5 <- cor.test(x[n.idx,1],x[n.idx,5],method = 'spearman')
    n7 <- cor.test(x[n.idx,1],x[n.idx,7],method = 'spearman')
    p5 <- cor.test(x[!n.idx,1],x[!n.idx,5],method = 'spearman')
    p7 <- cor.test(x[!n.idx,1],x[!n.idx,7],method = 'spearman') 
  }
  
  abun.tmp <- data.frame(disease=idx,feature.type=c('PR-decreasing','PR-decreasing','PR-increasing','PR-increasing'),
                         sample.type=c('case','ctrl','case','ctrl'),
                         rho=c(n6$estimate,n7$estimate,p6$estimate,p7$estimate),
                         p=c(n6$p.value,n7$p.value,p6$p.value,p7$p.value))
  prev.tmp <- data.frame(disease=idx,feature.type=c('PR-decreasing','PR-decreasing','PR-increasing','PR-increasing'),
                         sample.type=c('case','ctrl','case','ctrl'),
                         rho=c(n4$estimate,n5$estimate,p4$estimate,p5$estimate),
                         p=c(n4$p.value,n5$p.value,p4$p.value,p5$p.value))
  prev.cor <- rbind(prev.cor,prev.tmp)
  abun.cor <- rbind(abun.cor,abun.tmp)
}

prev.cor$type <- 'Prevalence'
abun.cor$type <- 'Abundance'

prev.abun.cor <- rbind(abun.cor,prev.cor)
prev.abun.cor$feature.sample <- paste(prev.abun.cor$feature.type, ' in ',prev.abun.cor$sample.type)
prev.abun.cor$signif <- ifelse(prev.abun.cor$p<0.05,'p<0.05','p≥0.05')

p <- ggplot(prev.abun.cor, aes(x = type, y = abs(rho))) +
  # 1. 小提琴图 - 浅色填充
  geom_violin(
    alpha = 0.3,           # 透明度
    color = "gray30",      # 边框颜色
    linewidth = 0.5,       # 边框线宽
    trim = TRUE,           # 不修剪边缘
    scale = "width",       # 宽度相同
    width = 0.8            # 宽度
  ) +
  
  # 2. 箱形图 - 黑线
  geom_boxplot(
    width = 0.15,          # 窄箱形图
    alpha = 0.8,           # 透明度
    color = "black",       # 黑线
    fill = "white",        # 白色填充
    outlier.shape = NA,    # 隐藏异常值（用抖动点显示）
    linewidth = 0.6        # 线宽
  ) +
  
  # 3. 抖动点 - 彩色点
  geom_quasirandom(
    aes(color = feature.type),     # 按类型着色
    size = 2,              # 点大小
    alpha = 0.6,           # 透明度
    width = 0.2,           # 抖动宽度
    varwidth = TRUE,       # 宽度可变
    method = "pseudorandom" # 伪随机排列
  ) +
  
  # 颜色设置
  scale_color_manual(
    values = c(
      "PR-decreasing" = "#8AB1D2", #"#C2D8EB", #"#8AB1D2",
      "PR-increasing" = "#E58579" #"#F2C7C2" #"#E58579"  新颜色淡30%
    ),
    name = ""  # 图例标题
  ) +
  
  # 坐标轴和标签
  scale_y_continuous(
    limits = c(0, 0.5),
    breaks = seq(0, 0.5, 0.1),
    expand = expansion(mult = c(0.05, 0.1))  # 扩展空间
  ) +
  
  labs(
    x = "",
    y = "Spearman correlation (rho)",
    title = "Importance vs. Abundance/Prevalence"
  ) +
  
  # 主题设置
  theme_minimal() +
  theme(
    plot.title = element_text(
      hjust = 0.5, 
      size = 16, 
      face = "bold",
      margin = margin(b = 15)
    ),
    axis.title.x = element_text(
      size = 13, 
      face = "bold",
      margin = margin(t = 10)
    ),
    axis.title.y = element_text(
      size = 13, 
      face = "bold",
      margin = margin(r = 10)
    ),
    axis.text = element_text(size = 11),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_line(color = "gray95", linewidth = 0.2),
    panel.grid.major.x = element_blank(),  # 移除垂直网格线
    legend.position ="right",
    legend.title = element_text(face = "bold", size = 11),
    legend.text = element_text(size = 10),
    plot.margin = margin(20, 20, 20, 20)
  ) #+ theme(legend.position = c(0.15, .97))

# 显示图形
print(p)
ggsave('importance.vs.abun.prev.pdf',p,width = 6,height = 6)

rank='.n.rank'
rank='.p.rank'

topN=30
show.effect.sign=T

bad <- biomarker.stat(features.info,'.p.rank',topN,c("#E53935","#FEE0D2"))
good <- biomarker.stat(features.info,'.n.rank',topN,c("#2E74B5","#DEEBF7"))
ggsave('shared_bad_biomarkers.pdf',bad[[2]],width = 4,height = 4.17) # pdf 3.97x4.17
ggsave('shared_good_biomarkers.pdf',good[[2]],width = 4,height = 4.17)

phylum.info <- features.info[unique(c(bad[[1]]$shared_taxa[bad[[1]]$shared_labels>0],good[[1]]$shared_taxa[good[[1]]$shared_labels>0])),'phylum',drop=F]

idx1 <- rowSums(features.info[,grepl('.p.rank',colnames(features.info))]<=topN,na.rm = T)>0
idx2 <- rowSums(features.info[,grepl('.n.rank',colnames(features.info))]<=topN,na.rm = T)>0

phylum.info <- features.info[idx1 | idx2,'phylum',drop=F]
x <- data.frame(table(phylum.info$phylum))
x <- x[order(x$Freq,decreasing = T),]
colnames(phylum.info) <- 'Phylum'
colors <- colorRampPalette(brewer.pal(10,'Set3'))(length(unique(phylum.info[,1])))
phylum.colors <- rep(colors,length.out=length(unique(phylum.info[,1])))
names(phylum.colors) <- x$Var1
annotation.colors <- list(Phylum=phylum.colors)
barplot(1:26,col=phylum.colors)

w30o3.p <- features.heatmap(features.info,annotation.colors,'.p.rank',topN = 30,prev = 3,show.effect.sign) # pdf landscape 8.84 x 7.08
w30o3.n <- features.heatmap(features.info,annotation.colors,'.n.rank',topN = 30,prev = 3,show.effect.sign) # pdf 7.65 x 6

p <- w30o3.p[[1]]
p.w <- w30o3.p[[2]][p$tree_row$order,p$tree_col$order] # 41
p.r <- w30o3.p[[3]][p$tree_row$order,p$tree_col$order] # 
colnames(p.r) <- colnames(p.w)

n <- w30o3.n[[1]]
n.w <- w30o3.n[[2]][rev(n$tree_row$order),p$tree_col$order] # y column use the same label as in p
n.r <- w30o3.n[[3]][rev(n$tree_row$order),p$tree_col$order] # 34, upside down
colnames(n.r) <- colnames(n.w)

# all(colnames(p.w)==colnames(n.w)) # TRUE
intersect(rownames(p.w),rownames(n.w)) # 0

w <- rbind(p.w,n.w)
r <- rbind(p.r,n.r)

selected.annotation.colors <- annotation.colors$Phylum[unique(features.info[rownames(w),'phylum'])]
selected.annotation.colors <- list(Phylum=selected.annotation.colors)
custom_breaks <- seq(-.5, .5, length.out = 100)
mat2_colors <- colorRampPalette(c("blue", "white", "red"))(100) 
mat2_colors <- colorRampPalette(c("#2E74B5", "white", "#E53935"))(100)
#mat2_colors <- colorRampPalette(c("#8AB1D2", "white", "#E58579"))(100) # kexin



pp <- pheatmap::pheatmap(w,cluster_rows = F,cluster_cols = F,display_numbers = r,
                         fontsize_number = 8, color = mat2_colors,fontsize_row = 8,
                         border_color = "gray90",
                         annotation_row = phylum.info,
                         annotation_colors = selected.annotation.colors,
                         breaks = custom_breaks) # pdf 8x9，portrait; 8.8 x 7.58 portrait

ggsave('top30.biomarkers.swap.pdf',pp,width = 7.58,height = 8.8)

sign.stat <- data.frame(PR_decreasing=c(NA,NA,NA),PR_increasing=c(NA,NA,NA))
sign.stat$Category <- c('+','-','ns')
sign.stat[1,1] <- sum(grepl('+',unlist(n.r),fixed = T))
sign.stat[2,1] <- sum(grepl('-',unlist(n.r),fixed = T))
sign.stat[3,1] <- sum(unlist(n.r)!='')-sign.stat[1,1]-sign.stat[2,1]
sign.stat[1,2] <- sum(grepl('+',unlist(p.r),fixed = T))
sign.stat[2,2] <- sum(grepl('-',unlist(p.r),fixed = T))
sign.stat[3,2] <- sum(unlist(p.r)!='')-sign.stat[1,2]-sign.stat[2,2]
data_long <- sign.stat %>%
  pivot_longer(
    cols = c(PR_decreasing, PR_increasing),
    names_to = "Group",
    values_to = "Count"
  )
data_long$Enriched <- factor(data_long$Category,levels=c('-','+','ns'),labels = c('Control enriched','Case enriched','NS'))
data_percent <- data_long %>%
  group_by(Group) %>%
  mutate(
    Percentage = Count / sum(Count) * 100,
    Label = sprintf("%.1f%%", Percentage)
  ) %>%
  ungroup()

p <- ggplot(data_percent, aes(x = Group, y = Count, fill = Enriched)) +
  geom_col(position = "dodge") +
  geom_text(
    aes(label = Label),
    position = position_dodge(width = 0.9),
    vjust = -0.5,
    size = 3.5
  ) +
  scale_fill_manual(
    values = c("Case enriched" = "#E58579",  # 红色
               "Control enriched" = "#8AB1D2",  # 蓝色
               "NS" = "gray75"),  # 绿色
    name=''
  ) +
  labs(y='Species count',
       x='',
       title='Shared Biomarker Enrichment Direction')+
  theme_minimal() +
  theme( panel.grid.major.x = element_blank(),
         plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
         legend.position = "right"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))
print(p)
ggsave('shared.biomarker.enrichment.pdf',p,width = 6,height = 5)




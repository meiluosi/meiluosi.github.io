# R语言项目实战

# 数据科学从业者调查

首先加载需要使用到的R包

```R
> library(data.table)# fread()
> library(tidyverse) # ggplot()
```

接下来就是读取数据文件，并且查看数据维度。

```R
responses <- fread("C:\\Users\\11603\\Documents\\kaggle-data-mining-master\\multipleChoiceResponses.csv")
dim(responses)
```

显示结果如下：

```R
[1] 16716   228
```

可以看出，在这个数据集中存在228个变量，一共有16716条样本数据。

但是我们不是所有变量都要用，因此筛选出16个变量。

```R
> responses <- responses[,.(Age,Country,CurrentJobTitleSelect,MLToolNextYearSelect,MLMethodNextYearSelect,EmploymentStatus, FormalEducation,CoursePlatformSelect,FirstTrainingSelect,Tenure,JobSatisfaction,LanguageRecommendationSelect,JobSkillImportanceR,JobSkillImportancePython,JobSkillImportanceSQL,JobSkillImportanceBigData)]
> dim(responses)
[1] 16716    16
```

进一步地，我们查看数据的数值情况

```R
> responses$Age%>%
+ unique()
 [1]  NA  30  28  56  38  46  35  22  43  33  20  27  26  54  58
[16]  24  39  49  25  21  51  34  41  32  53  29  36  23  48  37
[31]  63  40  31  59  44  47  19  50  68  16  42  60  18   0  62
[46]  57  72  13  55  52  17  15  69  11  70  65  45 100  14  64
[61]  80   6  61  66   1  10  67  73  71  74  75   3  77  76  79
[76]  99  12   4   2  94  83  78   9  82  98
```

但是这里面存在问题，可以看见Age存在0和100的情况，这显然就是收集数据时候，受访者胡乱回答的。下面的代码查询每个具体年龄受访者的观测位置。

```R
> responses$Age%>%
+ grep("^1$",x=.)
[1]  2177  5019 14894 16639
> responses[Age!= 1][,1:2]
         Age       Country
       <int>        <char>
    1:    30 United States
    2:    28        Canada
    3:    56 United States
    4:    38        Taiwan
    5:    46        Brazil
   ---                    
16377:    24         Other
16378:    25     Indonesia
16379:    25        Taiwan
16380:    16     Singapore
16381:    27         Japan
```

接下来使用ifelse的方法去将离谱的年龄给去除，这里去除了0-3岁和100岁的受访者。并且查看数据的分布情况。

```R
responses$Age <- ifelse(responses$Age %in% c(0:3, 100), 7, responses$Age)
responses[!is.na(Age)]$Age%>% 
+ unique()
 [1] 30 28 56 38 46 35 22 43 33 20 27 26 54 58 24 39 49 25 21 51
[21] 34 41 32 53 29 36 23 48 37 63 40 31 59 44 47 19 50 68 16 42
[41] 60 18  7 62 57 72 13 55 52 17 15 69 11 70 65 45 14 64 80  6
[61] 61 66 10 67 73 71 74 75 77 76 79 99 12  4 94 83 78  9 82 98
```

上面的工作完成了简单的数据清洗，我们仅仅只是针对年龄做了这样的过程，其实还存在其他字段需要处理，不过这里暂且不多做处理，只是简单说明存在这个过程就可以。

### 数据科学从业者数据探索性分析

```R
> df_country_age <- responses %>%
group_by(Country) %>% # 按照Country进行统计
summarise(AgeMedian = median(Age, na.rm = T)) %>% # 统计Age的中位数
arrange(desc(AgeMedian)) # 按照Age进行降序排列
> df_country_age[1:10,]
## # A tibble: 10 x 2
##Country AgeMedian
##<chr> <dbl>
##1 New Zealand39
##2 Spain37
##3 Ireland35
##4 Australia34
##5 Canada 34
##6 Denmark34
##7 Israel 34
##8 Italy34
##9 Netherlands34
## 10 Norway 34
```

上面的代码用以实现根据Age的值进行降序排列。

```R
> df_country_age <- df_country_age %>%
mutate(Country = ifelse(Country == "New Zealand", "新西兰", Country),
 Country = ifelse(Country == "Spain", "西班牙", 
ifelse(Country == "Ireland", "爱尔兰", Country)))
> df_country_age %>% 
head(10) %>%
ggplot(aes(x = reorder(Country, AgeMedian), y = AgeMedian,fill = Country)) +
geom_bar(stat = 'identity') +
labs(x = "", y = '年龄中位数') +
geom_text(aes(label = AgeMedian), hjust = 1.5, colour = 'white') +
coord_flip() +
theme_minimal() +
theme(legend.position = 'none') # 移除图例。
> df_country_age %>% 
tail(3) %>%
mutate(Country = case_when(Country == "Pakistan" ~ "巴基斯坦",
Country == "Indonesia" ~ "印度尼西亚",
Country == "Vietnam" ~ "越南")) %>% 
ggplot(aes(x = reorder(Country, AgeMedian), y = AgeMedian,fill = Country)) +
geom_bar(stat = 'identity') +
labs(x = "", y = '年龄中位数') +
geom_text(aes(label = AgeMedian), hjust = 1.5, colour = 'white') +
coord_flip() +
theme_minimal() +
theme(legend.position = 'none') 
```

这里选取不同国家可视化其年龄中位数

![image-20240915213959756](C:\Users\11603\OneDrive\个人项目\因果推断学习\工具学习\R语言学习\assets\image-20240915213959756.png)

![image-20240915214040626](C:\Users\11603\OneDrive\个人项目\因果推断学习\工具学习\R语言学习\assets\image-20240915214040626.png)

奇怪的是，巴基斯坦没有能够成功显示出其名称。

从上面的图可以获得一些信息。

> 1）新西兰受访者年龄中位数最大，这在一定程度上可以反映新西兰的受访者年龄偏大，如果进行进一步的推断，并以人口学数据佐证的话，可以间接推断出新西兰人口老龄化程度可能偏高。
>
> 2）印度尼西亚的受访者年龄中位数最小，这在一定程度上反应出印度尼西亚的受访者年龄偏小。
>
> 3）偏激一些的推断结论也可以总结为，发达国家中受访者的年龄中位数普遍高于发展中国家受访者的年龄中位数，但也可能因为其他因素的干扰，比如受访者的人数、接触到Kaggle的难易程度等，导致该结论并不可靠。

为了提升工作效率，下面对绘图的函数进行封装，后续就可以减少代码的输入

```R
> fun1 <- function(data, xlab, ylab, xname, yname) {
ggplot(data, aes(xlab, ylab)) +
geom_bar(aes(fill = xlab), stat = 'identity') +
labs(x = xname, y = yname) +
geom_text(aes(label = ylab), hjust = 1, colour = 'white') +
coord_flip() +
theme_minimal() +
theme(legend.position = 'none')
}
```

### 探索从业者职位

接下来我们的目标是探索美国受访者当中排名前三的职位。下面的代码实现了这一目标。

```R
> df_CJT_USA <- responses %>%
filter(CurrentJobTitleSelect != '' & Country == 'United States') %>% 
group_by(CurrentJobTitleSelect) %>%
summarise(Count = n()) %>%
arrange(desc(Count)) %>% 
mutate(CurrentJobTitleSelect = case_when(CurrentJobTitleSelect == "Data Scien-tist" ~ "数据科学家",
CurrntJobTitleSelect == "Software Developer/Software Engineer" ~ "软件开发/工程师",
CurrentJobTitleSelect == "Other" ~ "其他"))
> data <- head(df_CJT_USA, 3)
> xname <- ''
> yname <- '受访者数量'
> fun1(data, reorder(data$CurrentJobTitleSelect, data$Count), data$Count, xname, yname)
```

![image-20240915214644199](C:\Users\11603\OneDrive\个人项目\因果推断学习\工具学习\R语言学习\assets\image-20240915214644199.png)

同样的逻辑应用在其他国家，下面是新西兰的，改动的代码就是Country里面对应的字符串。

![image-20240915214912733](C:\Users\11603\OneDrive\个人项目\因果推断学习\工具学习\R语言学习\assets\image-20240915214912733.png)

我们可以简要得出，美国和新西兰两国的受访者职位排名前三的同为数据科学家、软件开发/工程师和其他。不过值得注意的是，因为新西兰受访人数过少，该结果可能并不准确。

### 未来将会使用机器学习工具

接下来的目标是探索美国数据科学从业者未来将会学习的机器学习工具。

```R
> df_MLT_USA <- responses %>% # 筛选出MLToolNextYearSelect不为空且为美国Kaggle的观测。
filter(MLToolNextYearSelect != '' & Country == 'United States') %>%
group_by(MLToolNextYearSelect) %>%
summarise(Count = n()) %>%
arrange(desc(Count))
> data <- head(df_MLT_USA, 3)
> xname <- '机器学习语言'
> yname <- '人数'
> fun1(data, reorder(data$MLToolNextYearSelect, data$Count), data$Count, xname, yname)
```

![image-20240915215145081](C:\Users\11603\OneDrive\个人项目\因果推断学习\工具学习\R语言学习\assets\image-20240915215145081.png)

新西兰的结果如下：

![image-20240915215301078](C:\Users\11603\OneDrive\个人项目\因果推断学习\工具学习\R语言学习\assets\image-20240915215301078.png)

> 我们可以得出如下两点结论。
>
> 1）美国受访者未来将会学习的最热门的工具是TensorFlow、Python和Spark/Millib。2）新西兰受访者希望学习的工具则为TensorFlow、R和Python。笔者猜测之所以R会成为新西兰第二热门的工具，很大原因可能是因为R诞生于新西兰的奥克兰大学，相较于美国，R在新西兰有比较良好的群众基础。

### 明年将要学习的机器学习方法

改变参数MLMethodNextYearSelect，查看明年美国的受访者要学习的机器学习算法。

```R
> df_MLM_USA <- responses %>% # 筛选出MLMethodNextYearSelect不为空且为美国Kaggle的观测。
filter(MLMethodNextYearSelect != '' & Country == 'United States') %>% 
group_by(MLMethodNextYearSelect) %>%
summarise(Count = n()) %>% 
arrange(desc(Count)) %>% 
mutate(MLMethodNextYearSelect = case_when(MLMethodNextYearSelect == "Deep learning" ~ "深度学习",
MLMethodNextYearSelect == "Neural Nets" ~ "神经网络",
MLMethodNextYearSelect == "Time Series Analysis" ~ "时间序列分析",
MLMethodNextYearSelect == "Bayesian Methods" ~ "贝叶斯方法",
MLMethodNextYearSelect == "Text Mining" ~ "文本挖掘"))
> data <- head(df_MLM_USA, 5)
> xname <- '机器学习方法'
> yname <- '人数'
> fun1(data, reorder(data$MLMethodNextYearSelect, data$Count), data$Count, xname, yname)
```

![image-20240915221544683](C:\Users\11603\OneDrive\个人项目\因果推断学习\工具学习\R语言学习\assets\image-20240915221544683.png)

```R
> df_MLM_NZ <- responses %>% # 筛选出MLMethodNextYearSelect不为空且为新西兰Kaggle的观测。
filter(MLMethodNextYearSelect != '' & Country == 'New Zealand') %>%
group_by(MLMethodNextYearSelect) %>%
summarise(Count = n()) %>%
arrange(desc(Count)) %>% 
mutate(MLMethodNextYearSelect = case_when(MLMethodNextYearSelect == "Deep learning" ~ "深度学习",
MLMethodNextYearSelect == "Neural 
Nets" ~ "神经网络",
MLMethodNextYearSelect == "Anomaly 
Detection" ~ "异常检测",
MLMethodNextYearSelect == "Genetic & Evolutionary Algorithms" ~ "遗传进化算法",
MLMethodNextYearSelect == "Time Series Analysis" ~ "时间序列分析"))
> data <- head(df_MLM_NZ, 5)
> xname <- '机器学习方法'
> yname <- '人数'
> fun1(data, reorder(data$MLMethodNextYearSelect, data$Count), data$Count, xname, yname)
```

![image-20240915221813745](C:\Users\11603\OneDrive\个人项目\因果推断学习\工具学习\R语言学习\assets\image-20240915221813745.png)



# 


# 安装 ISLR 包（只需第一次）
# install.packages("ISLR")  # 如已安装可注释掉

# 加载 ISLR 包
library(ISLR)

# 查看数据确认已加载
head(Auto)

# 将数据保存为 CSV 文件，保存在当前路径
write.csv(Auto, file = "Auto.csv", row.names = FALSE)

# 打印提示信息
cat("✅ Auto.csv has been saved to: ", getwd(), "\n")

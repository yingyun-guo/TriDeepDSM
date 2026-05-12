# 使用官方 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制环境配置文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有网站代码到容器中
COPY . .

# 暴露 Hugging Face 默认的 7860 端口
EXPOSE 7860

# 启动 Django 服务器 (绑定到 7860 端口)
CMD ["python", "manage.py", "runserver", "0.0.0.0:7860"]
# GitHub 同步指南

## 已完成的工作

✅ Git 仓库已初始化  
✅ MMPE 和 climate_analysis_toolkit 目录已添加到仓库  
✅ 初始提交已创建（61 个文件，55943 行代码）

## 接下来的步骤

### 1. 配置 Git 用户信息（如果还没有配置）

```bash
# 设置全局配置（推荐）
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"

# 或者只为当前仓库设置
cd /sas12t1/ffyan
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub邮箱"
```

### 2. 在 GitHub 上创建新仓库

1. 登录 GitHub
2. 点击右上角的 "+" 号，选择 "New repository"
3. 输入仓库名称（例如：`climate-analysis-tools`）
4. 选择 Public 或 Private
5. **不要**勾选 "Initialize this repository with a README"（因为我们已经有了代码）
6. 点击 "Create repository"

### 3. 添加远程仓库并推送

```bash
cd /sas12t1/ffyan

# 添加远程仓库（将 YOUR_USERNAME 和 REPO_NAME 替换为你的实际信息）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 或者使用 SSH（如果你配置了 SSH 密钥）
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# 推送代码到 GitHub
git branch -M main  # 将分支重命名为 main（GitHub 默认分支名）
git push -u origin main
```

### 4. 如果遇到认证问题

如果使用 HTTPS 推送时要求输入密码，你可能需要：

**选项 A：使用 Personal Access Token**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 生成新 token，勾选 `repo` 权限
3. 推送时使用 token 作为密码

**选项 B：使用 SSH 密钥**
1. 生成 SSH 密钥：`ssh-keygen -t ed25519 -C "your_email@example.com"`
2. 将公钥添加到 GitHub：Settings → SSH and GPG keys → New SSH key
3. 使用 SSH URL 添加远程仓库

### 5. 后续更新

以后如果有代码更改，使用以下命令同步：

```bash
cd /sas12t1/ffyan

# 查看更改
git status

# 添加更改的文件
git add MMPE/ climate_analysis_toolkit/

# 提交更改
git commit -m "描述你的更改"

# 推送到 GitHub
git push
```

## 注意事项

- `.gitignore` 已配置，会自动排除数据文件、输出目录、日志等
- 备份目录（MMPE_backup/、toolkit_backup/）已被排除
- 如果以后需要更新 Git 用户信息，使用上面的 `git config` 命令

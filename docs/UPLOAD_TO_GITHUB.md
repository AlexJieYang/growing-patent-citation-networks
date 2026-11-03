# 上传到 GitHub

建议仓库名为 `patent-network-growth`。解压后，目录中即为仓库源码。

## 网页上传

1. 在 GitHub 创建空仓库。
2. 进入解压后的 `patent-network-growth` 文件夹。
3. 将内部文件上传到仓库根目录，保持 Notebook、`prelib/`、`config.json` 同级。
4. 同时上传 `.github/`、`.gitignore`、`.gitattributes`；`.github/workflows/tests.yml` 提供自动测试。

压缩包只含源码。如果先本地运行再网页上传，不选择生成的 `outputs/`、`data/raw/`、`config.local.json` 或虚拟环境。

## 使用 Git

在项目目录运行，并将地址替换为自己的空仓库地址：

```bash
git init -b main
git add .
git commit -m "Add patent network research workflow"
git remote add origin https://github.com/YOUR_USER/YOUR_REPOSITORY.git
git push -u origin main
```

命令行推送需要先完成自己的 GitHub 身份验证，见 [官方说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 后续提交

本地数据和结果已在 `.gitignore` 中排除。Notebook 执行后，提交前运行：

```bash
python tools/clean_notebooks.py
python tools/clean_notebooks.py --check
git status --short
```

GitHub Actions 在 push、pull request 时运行，也可手动启动。单元测试不需要真实数据；集成测试生成合成输入并检查全部计算与绘图。作者与引用信息在 `CITATION.cff` 中，没有填入未知仓库地址或 DOI。

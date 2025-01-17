# College Hoops

## Conda environment

1. Create new conda environment
```
conda env create --name college-hoops
```
2. Add packages to conda
```
conda install anaconda::pandas
```
```
conda install -c anaconda beautifulsoup4
```
```
conda install anaconda::lxml
```
3. Set up jupyter for conda environment ([sauce](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook))

```
pip install jupyter ipykernel
```
```
python -m ipykernel install --user --name college-hoops --display-name "college-hoops"
```
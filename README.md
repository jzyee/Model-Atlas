# Charting and Navigating Hugging Face's Model Atlas
Official PyTorch Implementation for the "Charting and Navigating Hugging Face's Model Atlas" paper.  
<p align="center">
    🌐 <a href="https://horwitz.ai/model-atlas/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2503.10633" target="_blank">Paper</a> 
</p>

___

> **Charting and Navigating Hugging Face's Model Atlas**<br>
> Eliahu Horwitz, Nitzan Kurer, Jonathan Kahana, Liel Amar, Yedid Hoshen <br>
> <a href="https://arxiv.org/abs/2503.10633" target="_blank">https://arxiv.org/abs/2503.10633 </a> <br>
>
>**Abstract:** As there are now millions of publicly available neural networks, searching and analyzing large model repositories becomes increasingly important.
> Navigating so many models requires an *atlas*, but as most models are poorly documented charting such an atlas is challenging.
> To explore the hidden potential of model repositories, we chart a preliminary atlas representing the documented fraction of Hugging Face.
> It provides stunning visualizations of the model landscape and evolution. We demonstrate several applications of this atlas including predicting model attributes (e.g., accuracy),
>  and analyzing trends in computer vision models. However, as the current atlas remains incomplete, we propose a method for charting undocumented regions.
> Specifically, we identify high-confidence structural priors based on dominant real-world model training practices.
> Leveraging these priors, our approach enables accurate mapping of previously undocumented areas of the atlas.
> We publicly release our datasets, code, and interactive atlas.

___
**The current code is an initial rough version of the preprocessing code used to create the model atlas of the current state of Hugging Face. It cleans and organizes the data and generates graphml files that can be loaded into Gephi for visualization.**

**In the coming weeks I will publish a cleaner version of the code as well as the code to run the interactive model atlas localy and the code for our method for recovering the structure of unknown regions...**

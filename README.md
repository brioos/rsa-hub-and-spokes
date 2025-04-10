# rsa-hub-and-spokes

Ez a szakdolgozatomhoz és a TDK-hoz használt adatok és kódok repo-ja. A legtöbb kód nem saját gépen futott, hanem a HUN-REN TTK clusterén. 

Struktúra:  
├── correlations/       Különböző ROI-k neurális és szemantikus RDM-jének korrelációs eredményei (.pkl fájlok)  
├── figures/   
	  - ├── neural_rdm/         Neurális RDM-ek vizualizáció  
	  - └── result_figures/     Statisztikai elemzések ábrái  
├── masks/              ROI maszk (.nii.gz fájlok)  
    - ├── binary/             Bináris maszkok  
    - ├── func_masks/         Funkcionális térbe regisztrált maszkok  
    - └── probabilistic/      Valószínűségi maszkok (25%)  
├── rdms/               RDM-ek csv-ben kiexportálva  
├── scripts/            Elemzéshez használt kódok  
├── stimuli/            Kísérleti ingeranyagok (szópárak)   

from model import CycleGAN


for dataset_path in "cezanne2photo", "vangogh2photo":
    for norm in["L2", "SmoothL1Loss", "L1"]:
        # Load the model
        cycleGan = CycleGAN(dataset_path=dataset_path, norm=norm)

        # Eval the model
        cycleGan.make_gif()


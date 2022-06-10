

import os
import subprocess
import logging
logging.getLogger().setLevel(logging.INFO)

def export_image(filename, fig=None, type="pgf", **kwargs):
    ''' Export a Matplotlib figure to file
		type: Defines the type of the exported figure
			 - "pgf":exports the pgf with the corresponding pdf
			 - "all": does both
	'''

    # remove filename ending
    file = os.path.basename(filename)
    dir = os.path.dirname(filename)
    filename = os.path.join(dir, os.path.splitext(file)[0])

    if type == "pgf" or type == "all":
        fig.savefig(filename + ".pdf", **kwargs)
        logging.info("Saved image {}".format(os.path.abspath(filename+ ".pdf")))
        fig.savefig(filename + ".pgf", **kwargs)
    if type == "emf" or type == "all":
        fig.savefig(filename + ".svg", **kwargs)
        logging.info("Saved image {}".format(os.path.abspath(filename+ ".svg")))
    if type == "png":
        fig.savefig(filename + ".png", **kwargs)
        logging.info("Saved image {}".format(os.path.abspath(filename+ ".png")))



def convertToEmf(filename):
    dummy_ending = ".pdf"
    dummy_export_param = "--export-pdf"

    emf_ending = ".emf"
    emf_export_param = "--export-emf"

    # remove filename ending
    file = os.path.basename(filename)
    dir = os.path.dirname(filename)
    filename_emf = os.path.splitext(file)[0] + "_converted" + dummy_ending
    cmd = "inkscape" + " " + file + " " + dummy_export_param+"="+filename_emf
    os.chdir(dir)
    os.listdir()
    output = subprocess.call(cmd, shell=True)
    print(output)
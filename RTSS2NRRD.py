import os
from glob import glob
from typing import Dict

# https://github.com/acqxi/hnc/blob/main/makeData.py
# import getMask
# import lapAnno

SAVE_FOLDER = './RP-2019-2020/nrrd'


# %%
def read_data1( dicomsFolder: str, saveFolder: str = SAVE_FOLDER, debug: bool = False, fewTest: bool = False ):
    RTSSs = glob( dicomsFolder + '**/RS.*.dcm', recursive=True )
    # ANNOS = lapAnno.read_contour_txt( '../../../0826-LAP-shareData/20220101-中榮病人/contour.txt' )
    for i, RTSSpath in enumerate( RTSSs ):
        case_name = RTSSpath.split( os.sep )[ -2 ]

            try:
                tumors: Dict[ str, getMask.Tumor ] = getMask.get_tumors_from_dcm( RTSSpath, debug=debug, exten=0 )
                for lap_name, tumor in tumors.items():
                    _, lnm, ene = ANNOS[ case_name ][ lap_name ]
                    if debug:
                        print( case_name, lap_name, lnm, ene )
                    tumor.save(
                        '-'.join( map( str, [ case_name, lap_name, lnm, ene ] ) ),
                        slot=1,
                        folderName='noEx_crop',
                        saveFolder=saveFolder,
                    )
                if fewTest and i > 5:
                    break
            except KeyError as e:
                print( case_name, 'haven\'t contour data for', e )



read_data1( 'D:\\OneDrive\\CTC-研究-RP-Huang\\RP-2019-2020\\RP-2019-46p111ct' )


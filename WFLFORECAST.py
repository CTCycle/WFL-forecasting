import sys
import art

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_assets import UserOperations

# welcome message
#------------------------------------------------------------------------------
ascii_art = art.text2art('WFL Forecast')
print(ascii_art)

# [MAIN MENU]
#==============================================================================
# module for the selection of different operations
#==============================================================================
user_operations = UserOperations()
operations_menu = {'1' : 'Collect Win for Life extractions',
                   '2' : 'Analyze WFL data',  
                   '3' : 'Pretrain WFL model',
                   '4' : 'Pretrain WFL model (K-fold training)',
                   '5' : 'Predict next extraction',                                                     
                   '6' : 'Exit and close'}

while True:
    print('------------------------------------------------------------------------')    
    op_sel = user_operations.menu_selection(operations_menu)
    print()     
    if op_sel == 1:
        import modules.WFL_collector
        del sys.modules['modules.WFL_collector']
    elif op_sel == 2:
        import modules.WFL_analysis
        del sys.modules['modules.WFL_analysis']
    elif op_sel == 3:
        import modules.WFL_training
        del sys.modules['modules.WFL_training']
    elif op_sel == 4:
        import modules.WFL_kfold_training
        del sys.modules['modules.WFL_kfold_training']
    elif op_sel == 5:
        import modules.WFL_forecast
        del sys.modules['modules.WFL_forecast']
    elif op_sel == 6:
        break



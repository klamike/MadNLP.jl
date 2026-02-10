module MadNLPMOI

import MadNLP
import NLPModels
import ParametricNLPModels
import MathOptInterface as MOI
import MathOptInterface.Utilities as MOIU

function __init__()
    setglobal!(MadNLP, :Optimizer, Optimizer)
    return
end

include("MOI_utils.jl")
include("MOI_wrapper.jl")
include("MOI_parametric.jl")

end # module MadNLPMOI

from common.nets.model_conditional_diffusion_mixste_s2s_grand_linLift import ConditionalDiffusionMixSTES2SGRANDLinLift
from common.nets.model_conditional_diffusion_mixste_s2f_grand_linLift import ConditionalDiffusionMixSTES2FGRANDLinLift


def HPE_model(MODEL_NAME):
    models = {
        'ConditionalDiffusionMixSTES2SGRANDLinLift': ConditionalDiffusionMixSTES2SGRANDLinLift,
        'ConditionalDiffusionMixSTES2FGRANDLinLift': ConditionalDiffusionMixSTES2FGRANDLinLift,
    }

    return models[MODEL_NAME]
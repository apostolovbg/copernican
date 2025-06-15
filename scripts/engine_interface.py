"""Interface between compiled models and computational engines."""
# DEV NOTE (v1.5a): Initial placeholder for Phase 0.

def run_engine(engine_module, model_callables, sne_data, bao_data):
    """Send prepared callables and data to the selected engine."""
    # TODO: integrate with actual engine implementations
    return engine_module.execute(model_callables, sne_data, bao_data)

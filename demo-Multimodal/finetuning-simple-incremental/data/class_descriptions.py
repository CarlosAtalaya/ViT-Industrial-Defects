"""
Descripciones Textuales para Clases de Defectos Industriales
==============================================================

Descripciones optimizadas para fusión multimodal texto-visual.
Diseñadas para maximizar discriminación semántica entre clases similares.

Prioridad: Diferenciar ROTURA vs RAYONES (principal confusión semántica).
"""

CLASS_DESCRIPTIONS = {
    "NORMAL": {
        "id": 0,
        "description": "a clean metal surface without any visible defects or damage",
        "keywords": ["clean", "intact", "flawless", "undamaged"],
        "semantic_focus": "absence of defects"
    },
    
    "PERFORACIONES": {
        "id": 1,
        "description": "a circular hole or perforation penetrating through the metal surface",
        "keywords": ["hole", "circular", "penetrating", "drilled"],
        "semantic_focus": "deep circular penetration"
    },
    
    "RAYONES_ARANAZOS": {
        "id": 2,
        "description": "a thin superficial scratch on the metal surface without depth or penetration",
        "keywords": ["scratch", "superficial", "surface", "thin", "shallow"],
        "semantic_focus": "superficial linear damage"
    },
    
    "DEFORMACIONES": {
        "id": 3,
        "description": "a bent or deformed metal surface showing structural distortion or warping",
        "keywords": ["bent", "deformed", "warped", "distorted"],
        "semantic_focus": "structural shape change"
    },
    
    "CONTAMINACION": {
        "id": 4,
        "description": "dirt stains or foreign material contaminating the metal surface",
        "keywords": ["dirt", "stain", "contaminated", "foreign material"],
        "semantic_focus": "surface contamination"
    },
    
    "ROTURA_FRACTURA": {
        "id": 5,
        "description": "a deep crack or fracture breaking through the metal with visible depth and penetration",
        "keywords": ["crack", "fracture", "deep", "breaking", "penetrating"],
        "semantic_focus": "deep structural break"
    }
}


def get_text_prompts(template="A defect showing {}"):
    """
    Genera prompts de texto para encoder CLIP.
    
    Args:
        template: Plantilla de prompt (debe contener {} para descripción)
    
    Returns:
        list: Lista de 6 prompts textuales ordenados por class_id
    """
    prompts = []
    
    # Ordenar por id para mantener consistencia
    sorted_classes = sorted(CLASS_DESCRIPTIONS.items(), key=lambda x: x[1]['id'])
    
    for class_name, cls_info in sorted_classes:
        prompt = template.format(cls_info['description'])
        prompts.append(prompt)
    
    return prompts


def get_class_embeddings_info():
    """
    Retorna información estructurada para análisis.
    
    Returns:
        dict: Información por clase con focus semántico
    """
    info = {}
    for class_name, cls_info in CLASS_DESCRIPTIONS.items():
        info[cls_info['id']] = {
            'name': class_name,
            'description': cls_info['description'],
            'semantic_focus': cls_info['semantic_focus'],
            'keywords': cls_info['keywords']
        }
    return info


def print_descriptions():
    """Imprime todas las descripciones para verificación."""
    print("\n" + "="*70)
    print("DESCRIPCIONES TEXTUALES - FASE 2 MULTIMODAL")
    print("="*70)
    
    for class_name, cls_info in sorted(CLASS_DESCRIPTIONS.items(), key=lambda x: x[1]['id']):
        print(f"\n[{cls_info['id']}] {class_name}")
        print(f"    Description: {cls_info['description']}")
        print(f"    Focus: {cls_info['semantic_focus']}")
        print(f"    Keywords: {', '.join(cls_info['keywords'])}")
    
    print("\n" + "="*70)
    print("PROMPTS GENERADOS:")
    print("="*70)
    for i, prompt in enumerate(get_text_prompts()):
        print(f"[{i}] {prompt}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test de verificación
    print_descriptions()
    
    # Verificar que todas las clases tienen id único
    ids = [info['id'] for info in CLASS_DESCRIPTIONS.values()]
    assert len(ids) == len(set(ids)), "IDs duplicados detectados!"
    assert set(ids) == set(range(6)), "IDs deben ser 0-5"
    
    print("✅ Verificación completa: 6 clases con descripciones únicas")
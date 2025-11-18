
class PromptTemplates:
    @staticmethod
    def get_relation_prompt(labels, prompt_type="base"):
        
        base_template = "Describe the visual relation between the two entities: "+labels[0]+" and "+labels[1]+" in the image. The relation is directed, with "+labels[0]+" being the subject and "+labels[1]+" being the object of the relation. Based on your observations, summarize the visual relationship between "+labels[0]+" and "+labels[1]+" and describe it in the format: <sub>"+labels[0]+"</sub> <rel>[relationship]</rel> <obj>"+labels[1]+"</obj>."
            
        CoT = "Describe the visual relation between the two entities: "+labels[0]+" and "+labels[1]+" in the image. The relation is directed, with "+labels[0]+" being the subject and "+labels[1]+" being the object of the relation. Follow these steps: \n \n 1. **Entity Identification**: Identify and describe the visual characteristics of "+labels[0]+" and "+labels[1]+" in the image (e.g., shape, size, position, and notable features). \n 2. **Spatial Context**: Examine how "+labels[0]+" and "+labels[1]+" are positioned relative to each other. Consider the distance, orientation, and any overlaps or proximities. \n 3. **Functional Context**: Determine if there is any functional interaction between the two entities. Does "+labels[0]+" affect, support, or interact with "+labels[1]+" in some way? \n 4. **Final Relationship**: Based on your observations, summarize the visual relationship between "+labels[0]+" and "+labels[1]+" and describe it in the format: <sub>"+labels[0]+"</sub> <rel>[relationship]</rel> <obj>"+labels[1]+"</obj>."

        CoT_spatial = """
            Analyze the provided image and the two annotated bounding boxes. Focus only on the direct relative position between the two entities. Follow these steps:\n
            \n
            1. **Entity Position**: Briefly describe the location of each entity in the image (e.g., top-left, center-right).\n
            \n
            2. **Predicate Selection**: Determine the most accurate direct relative position between the two entities. Use only one of the following predicates:\n
            - **Below**: Entity 1 is completely or partially below Entity 2 along the vertical axis.\n
            - **Above**: Entity 1 is completely or partially above Entity 2 along the vertical axis.\n
            - **To the right of**: Entity 1 is horizontally positioned to the right of Entity 2, with no significant vertical overlap.\n
            - **To the left of**: Entity 1 is horizontally positioned to the left of Entity 2, with no significant vertical overlap.\n
            - **Behind**: Entity 1 appears farther back in depth relative to Entity 2 (based on image perspective or occlusion).\n
            - **In front of**: Entity 1 appears closer in depth relative to Entity 2 (based on image perspective or occlusion).\n
            \n
            3. **Reasoning**: Provide a concise explanation for your choice, considering both spatial arrangement and perspective.\n
            \n
            4. **Consistency Check**: Double-check that the selected predicate aligns with the spatial arrangement described in Step 1.\n
            \n
            Finally, summarize the spatial relationship between the two entities in the format: <rel>[spatial relationship]</rel>. For example: 1_person <rel>to the left of</rel> 2_dog. In this image, the entities are """+labels[0]+" and "+labels[1]+"."

        CoT_template = "Analyze and describe the directed visual relationship between two entities in an image, focusing on Entity 1 as the subject and Entity 2 as the object. Entity 1 is represented by the blue bounding box, while Entity 2 is the red bounding box. Follow these refined steps:\n\n1. **Entity Identification**  \n   - Clearly identify each entity's visual characteristics within the image. Note attributes such as shape, size, and position, alongside any distinguishing features.\n\n2. **Spatial Context**  \n   - Analyze the positioning of the entities relative to each other. Consider aspects like distance, orientation, and whether entities overlap or are close in proximity.\n\n3. **Functional Context**  \n   - Assess any potential functional interactions. Identify actions or roles that may indicate how Entity 1 impacts or interacts with Entity 2.\n\n4. **Integrative Reasoning**  \n   - Determine whether the relationship is predominantly spatial or functional. Use the analysis from steps 1-3 to support your reasoning and articulate it in a concise sentence.\n\nFinally, summarize the visual relationship in the format: `<sub>Entity 1</sub> <rel>relationship</rel> <obj>Entity 2</obj>`. Example: `<sub>1_person</sub> <rel>holding</rel> <obj>2_phone</obj>`.\n\n# Output Format\n\n- Provide your response as a structured sentence summarizing the relationship, followed by the formatted statement. \n\n# Notes\n\n- Ensure that the reasoning provided is comprehensive and ties together observations from all steps.\n- Consider both tangible interactions and abstract spatial nuances when formulating the result. \n Now, predict <rel></rel> for <sub>"+labels[0]+"</sub> and <obj>"+labels[1]+"</obj> based on this image."

        return {
            "base": base_template,
            "cot": CoT,
            "cot_spatial": CoT_spatial,
            "cot_refined": CoT_template
        }[prompt_type]
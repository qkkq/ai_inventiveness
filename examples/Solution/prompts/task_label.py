"""
This is a prompt for the task of converting text into bullet points as a Python list.
"""
from pydantic import BaseModel


class TaskLabelPrompt:
    """A prompt class for finding solution to problem using each Inventive Principle"""

    # -----------------------------------------------------------------------------
    # System Prompt
    # -----------------------------------------------------------------------------

    SYSTEM_PROMPT = """
    [Find solutions by using each Inventive Principle given by user]

    <prompt_objective> 
    Describe solution to given problem using single Inventive Principle.  
    </prompt_objective>  

    <prompt_rules>  
    - Generate single solution for each given Inventive Principle to the same problem
    - Use technical solutions and technical language
    - Create exactly one solution for one Inventive Principle.
    - do not change principle name
    - do not use more than one inventive principle in a solution 
    - do not add any additional text besides the solutions
    - after generating solution for each principle, provide the solution in the same order as the principles are given
    - after generating solution for each principle, check if it is present in message given by user. if not delete the solution
    </prompt_rules>  

    <prompt_examples>  

--------------------------------------------------------------------------------
    User: 
        description: A car trailer manufacturer has introduced a new lightweight composite material for the trailer bed, reducing the 
        overall weight by 25%. This weight reduction improves fuel efficiency and increases the payload capacity. The composite material,
        while durable, is more susceptible to UV degradation than traditional steel, reducing the expected lifespan of the trailer bed by 
        20% without additional protective coatings.",
        Principles:
            "Segmentation",
            "Parameter changes",
            "Universality",
            "Cheap short-living objects"
    AI: 
    Segmentation
    Instead of using a single large composite sheet for the trailer bed, divide it into modular, replaceable panels. This way, only the most UV-exposed or damaged sections need replacement over time, extending the overall lifespan while maintaining weight savings.

    Parameter Changes
    Modify the composite material by incorporating UV-resistant additives or a protective coating that enhances its durability. Alternatively, adjust the material composition to balance lightweight properties with improved UV resistance.

    Universality
    Design the trailer bed to serve multiple functions, such as integrating a UV-resistant protective layer that also improves grip or drainage. A multi-functional approach minimizes degradation while maintaining usability.

    Cheap Short-Living Objects
    Accept the limited lifespan but reduce costs by using a less expensive, easily replaceable composite layer. A thin, disposable UV-protective film or replaceable outer coating can be periodically swapped out without compromising the trailer's weight or functionality.
----------------------------------------------------------------------------------
    User:
description: A car trailer manufacturer has introduced a new lightweight composite material for the trailer bed, reducing the overall weight by 25%. 
This weight reduction improves fuel efficiency and increases the payload capacity. The composite material, while durable, is more susceptible to UV 
degradation than traditional steel, reducing the expected lifespan of the trailer bed by 20% without additional protective coatings.
Principles: 
            "Segmentation",
            "Parameter changes",
            "Universality",
            "Cheap short-living objects"
            
AI:
Segmentation
Instead of making the entire trailer bed from a single piece of composite material, divide it into smaller, modular panels. 
This way, only the most UV-exposed or damaged sections need replacement over time, rather than the entire bed, effectively 
increasing the overall lifespan while keeping weight reduction benefits.

Parameter Changes
Modify the composite material by introducing UV-resistant additives, coatings, or protective laminates to improve resistance to sunlight exposure. 
Alternatively, changing the surface texture or color to a more UV-reflective variant can also help reduce degradation.

Universality
Design the trailer bed with a built-in dual-function protective layer, such as a UV-resistant film that also provides anti-slip properties. 
Another approach is to use a removable, multi-purpose cover that not only protects against UV exposure but can also serve as a weatherproof shield.

Cheap Short-Living Objects
Accept that the composite material has a shorter lifespan and implement a cost-effective approach by using easily replaceable, 
low-cost composite panels. These panels could be designed for quick swap-out when they degrade, ensuring minimal downtime and maintenance costs.
--------------------------------------------------------------------------------
    User: 
        description: 
            "Engineers have redesigned a screw conveyor for the food industry, incorporating a polished stainless 
            steel auger to improve hygiene and reduce product adhesion. The new design has decreased cleaning time between batches 
            by 40% and reduced product waste by 15%. The smoother surface of the auger has led to a 10% decrease in conveying 
            efficiency, requiring longer operation times to transport the same volume of material."
        Principles:
            "Mechanics substitution",
            "Preliminary action",
            "Parameter changes",
            "Feedback"
    
    AI: 
        Mechanics Substitution
        Instead of relying solely on a screw conveyor, integrate an alternative conveying mechanism, 
        such as a vibratory conveyor or an air-assisted transport system. These alternatives could help 
        maintain or improve hygiene while minimizing the reduction in conveying efficiency.

        Preliminary Action
        Pre-treat the material before it enters the conveyor, such as by reducing its moisture content 
        or pre-coating it with a non-stick agent. This would decrease adhesion, helping to retain the 
        hygiene benefits while reducing the loss of efficiency.

        Parameter Changes
        Adjust the pitch or diameter of the auger to compensate for the loss in efficiency. Increasing 
        the pitch angle slightly or modifying the speed of the screw could improve the transport rate while maintaining the polished surface.

        Feedback
        Implement a real-time monitoring system to dynamically adjust conveyor speed based on load conditions. 
        Sensors could detect material flow and automatically adjust the auger speed to optimize efficiency while maintaining hygiene benefits.
        
    </prompt_examples> 
    """

    # -----------------------------------------------------------------------------
    # Pydantic Model
    # -----------------------------------------------------------------------------

    class OutputModel(BaseModel):
        """
        Pydantic model for the answers.
        """
        label: str

    # -----------------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------------

    @classmethod
    def get_messages(cls) -> list[dict]:
        """
        Get the messages.
        """
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": "{{text}}"},
        ]

    @classmethod
    def compile_messages(cls, text: str) -> list[dict]:
        """
        Compile the messages with the given context by replacing template variables.

        Args:
            text: The text content for the user message

        Returns:
            list[dict]: List of message dictionaries with populated template variables
        """
        messages = cls.get_messages()

        # Replace template variables in user message
        messages[1]["content"] = messages[1]["content"].replace("{{text}}", text)

        return messages

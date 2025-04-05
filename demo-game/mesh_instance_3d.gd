extends MeshInstance3D

var model: DiffusionModelContext
var txt2img_node: DiffusionImageGenerator

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	model = get_node("../DiffusionModelContext")
	model.start_model_build()

func _on_diffusion_model_context_build_complete() -> void:
	# start txt2img
	txt2img_node = get_node("DiffusionImageGenerator")
	txt2img_node.start_txt2img_worker()

func _on_diffusion_image_generator_complete(images: Array[Image]) -> void:
	var texture = ImageTexture.create_from_image(images[0])
	var material = StandardMaterial3D.new()
	material.albedo_texture = texture
	material_override = material

func _on_diffusion_model_context_step(step: int, steps: int, _time: float) -> void:
	var percent_complete = step/float(steps)
	var material = StandardMaterial3D.new()
	material.albedo_color = Color(percent_complete,percent_complete,percent_complete)
	material_override = material

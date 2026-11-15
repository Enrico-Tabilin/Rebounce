extends Node2D

var debounce = false

var albumCovers = {
	1 : ["res://UI/AlbumCovers/birdbrain.png"],
	2 : ["res://UI/AlbumCovers/ButterBTS.png"],
	3 : ["res://UI/AlbumCovers/flowerman.png"],
	4 : ["res://UI/AlbumCovers/gummy_bear.png"],
	5 : ["res://UI/AlbumCovers/mozart.png"],
	6 : ["res://UI/AlbumCovers/murderinmymind.png"],
	7 : ["res://UI/AlbumCovers/tATu.png"],
	8 : ["res://UI/AlbumCovers/taylor_swift.png"],
	9 : ["res://UI/AlbumCovers/twistedsisters.png"],
	10 : ["res://UI/AlbumCovers/verity.png"],
}

var song = {
	1 : ["res://songs/BIRDBRAIN By Jamie Paige"],
	2 : ["res://songs/BTS Butter"],
	3 : ["res://songs/Flower Man Toby Fox & CamelliaOfficial"],
	4 : ["res://songs/The Gummy Bear Song"],
	5 : ["res://songs/Mozart's Symphony no 40 - 1st movement"],
	6 : ["res://songs/KORDHELL - MURDER IN MY MIND"],
	7 : ["res://songs/t.A.T.u. - Я сошла с ума"],
	8 : ["res://songs/Taylor Swift - Blank Space"],
	9 : ["res://songs/I Wanna Rock by Twisted Sisters"],
	10 : ["res://songs/It's me, It's Verity - Horror Skunx"]
}

var desc = {
	1 : "Jamie Paige - BIRDBRAIN (Explicit)",
	2 : "BTS - Butter",
	3 : "Toby Fox - Flower Man",
	4 : "Gummibär - The Gummy Bear Song",
	5 : "Mozart - Symphony No. 40 - 1st Movement",
	6 : "KORDHELL - MURDER IN MY MIND",
	7 : "t.A.T.u. - Я сошла с ума",
	8 : "Taylor Swift - Blank Space",
	9 : "Twisted Sisters - I Wanna Rock",
	10 : "Horror Skunx - It's me, It's Verity"
}

var albumIndex = 1

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
	
	
func _on_play_pressed() -> void:
	$Title.visible = false
	$Play.visible = false
	$Play.disabled = true
	indexChanged()
	$Vinyl.visible = true
	$AlbumFrame.visible = true
	$Cover.visible = true
	$Left.visible = true
	$Right.visible = true
	$Desc.visible = true
	$MusicPlay.visible = true
	pass


func _on_left_pressed() -> void:
	if !debounce:
		debounce = true
		if albumIndex - 1 > 0:
			albumIndex -= 1
		else:
			albumIndex = albumCovers.size()
		indexChanged()
		var tween = get_tree().create_tween()
		tween.tween_property($Vinyl, "rotation", $Vinyl.rotation-PI/2, .5)
		tween.EASE_IN
		await get_tree().create_timer(.5).timeout
		debounce = false
	pass
	
func _on_right_pressed() -> void:
	if !debounce:
		debounce = true
		if albumIndex + 1 < albumCovers.size():
			albumIndex +=1
		else:
			albumIndex = 1
		indexChanged()
		var tween = get_tree().create_tween()
		tween.tween_property($Vinyl, "rotation", $Vinyl.rotation+PI/2, .5)
		tween.EASE_IN
		await get_tree().create_timer(.5).timeout
		debounce = false
	pass

func indexChanged() -> void:
	var tempCover:Texture2D = load(albumCovers[albumIndex][0])
	$Cover.texture = tempCover
	var tempSong:AudioStreamMP3 = load(song[albumIndex][0]+".mp3")
	$Preview.stream = tempSong
	$Preview.play()
	$Desc.text = desc[albumIndex]
	pass

func _on_musicplay_pressed() -> void:
	
	pass

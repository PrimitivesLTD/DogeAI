#pragma once

#include "CoreMinimal.h" // Unreal Engine Imports
#include "GameFramework/Character.h"
#include "ImageUtils.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/SceneCapture2D.h"
#include "Animation/AnimMontage.h"
#include "GameFramework/Actor.h"
#include "Doge.generated.h"

UCLASS()
class ADoge : public ACharacter
{
	GENERATED_BODY()

public:
	ADoge();
	USceneCaptureComponent2D* sceneCapture;
	UTextureRenderTarget2D* renderTarget;
	UAnimMontage* AnimRun;
	UAnimMontage* AnimTurnRight;
	UAnimMontage* AnimTurnLeft;
	UAnimMontage* AnimTurnRight90;
	UAnimMontage* AnimTurnLeft90;
	UAnimMontage* AnimRunFast;
	UAnimMontage* AnimLie;
	UAnimMontage* AnimJump;
	int n_natur = 0;
	int max_samples = 0;
	bool pretrain = false; //true - pretrain before running NaturAL (use on first cold run)
	float timer = 0.0f;
	float input_data[1][6][128][128];
	float output[1][9];
	float featsY[1][992];
	float featsZ[1][992];

	float crashInp[1][3][128][128];
	double featsEmb[1984];
	float crashReward[1][1];
	double* outNaturAL = nullptr;
	int preVal = 0;
	int argNaturAL = 0;
	int argmaxY = 0;
	int argmaxZ = 0;
	float temporary = 0;
	float max = 0;

protected:
	virtual void BeginPlay() override;

public:	
	virtual void Tick(float DeltaTime) override;

	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

};

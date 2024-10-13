import { IsString, IsNumber } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class CreateVideoDto {
  @ApiProperty({ description: 'User ID of the person who uploaded the video' })
  @IsString()
  userId: string;

  @ApiProperty({ description: 'URL of the uploaded video' })
  @IsString()
  videoUrl: string;

  @ApiProperty({ description: 'Size of the video file in bytes' })
  @IsNumber()
  fileSize: number;
}
